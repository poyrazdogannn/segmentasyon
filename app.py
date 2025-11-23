import base64
import io
import math
import os

from flask import Flask, render_template, request

import numpy as np
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

import torch
import segmentation_models_pytorch as smp
import requests  # 🔹 yeni

# ==========================
#  TEMEL AYARLAR
# ==========================
IMG_SIZE = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGENET_MEAN_ARR = np.array(IMAGENET_MEAN, dtype=np.float32)
IMAGENET_STD_ARR  = np.array(IMAGENET_STD , dtype=np.float32)

THR = 0.5            # Sabit threshold
AMP = True           # GPU varsa amp kullanılabilir

MODEL_PATH = "best_model.pth"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1GUr88MQG-73fdniaVwfg9buCHnVBppIt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
#  MODEL İNDİRME
# ==========================
def download_model_if_needed():
    """best_model.pth yoksa Google Drive'dan indirir."""
    if os.path.exists(MODEL_PATH):
        print("[INFO] Model dosyası zaten mevcut.")
        return
    if not MODEL_URL:
        raise RuntimeError("MODEL_URL boş! Model indirilemiyor.")
    print(f"[INFO] Model indiriliyor: {MODEL_URL}")
    r = requests.get(MODEL_URL, stream=True)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("[INFO] Model indirildi.")


# ==========================
#  YARDIMCI FONKSİYONLAR
# ==========================

def is_dicom(filename: str) -> bool:
    return filename.lower().endswith(".dcm")

def read_dicom_rgb_from_bytes(file_bytes: bytes):
    """DICOM'u RGB 0-255 ve PixelSpacing ile oku."""
    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    arr = ds.pixel_array.astype(np.float32)
    arr = apply_voi_lut(arr, ds)
    mn, mx = arr.min(), arr.max()
    arr = (arr - mn) / (mx - mn + 1e-8)
    img = np.stack([arr] * 3, axis=-1)
    spacing = (0.5, 0.5)
    if "PixelSpacing" in ds:
        spacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))
    return (img * 255).astype(np.uint8), spacing

def read_image_rgb_from_bytes(file_bytes: bytes):
    """PNG/JPG/BMP vb. RGB 0-255 ve default spacing ile oku."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Görsel okunamadı.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    spacing = (0.5, 0.5)  # non-DICOM için varsayılan
    return rgb, spacing

def preprocess_letterbox_center(img, device):
    """Train kodundaki center letterbox + normalize."""
    H0, W0 = img.shape[:2]
    s = IMG_SIZE / max(H0, W0)
    newH, newW = int(round(H0 * s)), int(round(W0 * s))
    r = cv2.resize(img, (newW, newH))

    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    pad_t = (IMG_SIZE - newH) // 2
    pad_l = (IMG_SIZE - newW) // 2
    canvas[pad_t:pad_t+newH, pad_l:pad_l+newW] = r

    x = canvas.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN_ARR) / IMAGENET_STD_ARR
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    meta = dict(H0=H0, W0=W0, newH=newH, newW=newW, pad_t=pad_t, pad_l=pad_l)
    return x, meta

def inverse_letterbox_to_original(prob_canvas, meta):
    pt, pl = meta["pad_t"], meta["pad_l"]
    newH, newW = meta["newH"], meta["newW"]
    H0, W0 = meta["H0"], meta["W0"]
    crop = prob_canvas[pt:pt+newH, pl:pl+newW]
    prob_orig = cv2.resize(crop, (W0, H0), interpolation=cv2.INTER_LINEAR)
    return prob_orig

def tta_predict_prob(model, x, device):
    """4’lü basit TTA (flip kombinasyonları)."""
    def infer(z):
        with torch.cuda.amp.autocast(enabled=AMP):
            return torch.sigmoid(model(z)).float()

    p0 = infer(x)
    p1 = torch.flip(infer(torch.flip(x, [-1])), [-1])
    p2 = torch.flip(infer(torch.flip(x, [-2])), [-2])
    p3 = torch.flip(infer(torch.flip(torch.flip(x, [-1]), [-2])), [-1, -2])

    prob = ((p0 + p1 + p2 + p3) / 4.0)[0, 0].detach().cpu().numpy()
    return prob

def measure_components(mask, spacing=(0.5, 0.5)):
    """Bağlı bileşenleri bul, alan / çevre / eş. çap hesapla."""
    rows = []
    dy, dx = spacing
    num, lab, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8
    )
    for i in range(1, num):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 3:
            continue
        area_mm = a * dy * dx
        x, y, w, h = stats[i, :4]
        cnts, _ = cv2.findContours(
            (lab == i).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        perim = cv2.arcLength(cnts[0], True) * (dx + dy) / 2
        eq_d = math.sqrt(4 * area_mm / math.pi)
        cx, cy = x + w // 2, y + h // 2
        rows.append(
            dict(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                cx=int(cx),
                cy=int(cy),
                area_mm2=float(area_mm),
                perim_mm=float(perim),
                eq_d_mm=float(eq_d),
            )
        )
    return rows

def create_overlay(img_rgb, mask_bin, comps):
    """Kontur + bbox + eş. çap etiketi ile overlay oluştur."""
    vis = img_rgb.copy()
    cnts, _ = cv2.findContours(
        mask_bin.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
    for c in comps:
        x, y, w, h = c["x"], c["y"], c["w"], c["h"]
        cx, cy = c["cx"], c["cy"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
        cv2.putText(
            vis,
            f"{c['eq_d_mm']:.1f} mm",
            (x, max(y - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2,
        )
    return vis

def run_inference_on_bytes(model, device, file_bytes: bytes, filename: str, thr: float):
    """Tek fonksiyon: bytes al → prob map → mask → ölçüm + overlay döndür."""
    if is_dicom(filename):
        img_rgb, spacing = read_dicom_rgb_from_bytes(file_bytes)
    else:
        img_rgb, spacing = read_image_rgb_from_bytes(file_bytes)

    H0, W0 = img_rgb.shape[:2]

    x, meta = preprocess_letterbox_center(img_rgb, device)
    prob_canvas = tta_predict_prob(model, x, device)
    prob_orig = inverse_letterbox_to_original(prob_canvas, meta)

    mask_bin = (prob_orig >= thr).astype(np.uint8)
    comps = measure_components(mask_bin, spacing=spacing)

    overlay_rgb = create_overlay(img_rgb, mask_bin, comps)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", overlay_bgr)
    if not ok:
        raise ValueError("Overlay encode edilemedi.")
    return buf, comps


# ==========================
#  MODEL YÜKLE
# ==========================

def build_model(weights_path=MODEL_PATH):
    print("[INFO] Model yükleniyor...")
    model = smp.UnetPlusPlus(
        encoder_name="timm-efficientnet-b3",  # train script ile aynı
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {weights_path}")
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print("[INFO] Model yüklendi.")
    return model


app = Flask(__name__)

# Önce model yoksa indir, sonra yükle
download_model_if_needed()
model = build_model(MODEL_PATH)

print("[INFO] Device:", DEVICE)
print(f"[INFO] Kullanılan eşik (THR): {THR:.2f}")


# ==========================
#  ROUTES
# ==========================

@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        overlay_image=None,
        measurements=None,
        error=None,
        filename=None,
        thr=f"{THR:.2f}",
    )

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template(
            "index.html",
            overlay_image=None,
            measurements=None,
            error="Dosya seçmediniz.",
            filename=None,
            thr=f"{THR:.2f}",
        )

    file = request.files["file"]
    if file.filename == "":
        return render_template(
            "index.html",
            overlay_image=None,
            measurements=None,
            error="Dosya adı boş.",
            filename=None,
            thr=f"{THR:.2f}",
        )

    file_bytes = file.read()
    try:
        buf, comps = run_inference_on_bytes(
            model=model,
            device=DEVICE,
            file_bytes=file_bytes,
            filename=file.filename,
            thr=THR,
        )
        overlay_b64 = base64.b64encode(buf).decode("utf-8")
        return render_template(
            "index.html",
            overlay_image=overlay_b64,
            measurements=comps,
            error=None,
            filename=file.filename,
            thr=f"{THR:.2f}",
        )
    except Exception as e:
        return render_template(
            "index.html",
            overlay_image=None,
            measurements=None,
            error=f"Hata: {e}",
            filename=None,
            thr=f"{THR:.2f}",
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

