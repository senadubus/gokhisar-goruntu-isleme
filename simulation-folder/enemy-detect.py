import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# AYARLAR
# =========================
MODEL_PATH = "best.pt"          # kendi YOLO weight dosyan
IMAGE_DIR = "images"            # 8 görselin olduğu klasör
OUTPUT_DIR = "outputs"          # sonuçların kaydedileceği klasör

CONF_THRES = 0.35
IOU_THRES = 0.45

PATCH_RATIO = 0.20              # bbox w,h'nin %20'si kadar patch
IGNORE_ORANGE = True            # turuncu balonu ignore et
MIN_VALID_PIXELS = 20           # patch içinde maske sonrası en az piksel
RED_MARGIN = 0.05               # enemy kararında kırmızı baskınlık marjı
BLUE_MARGIN = 0.05              # friend kararında mavi baskınlık marjı

# aynı class'ta sadece 2 örnek varsa outlier kararı anlamsız olabilir.
# o yüzden hem outlier hem de renk skoru birlikte kullanılıyor.
USE_OUTLIER_LOGIC = True

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# YARDIMCI FONKSİYONLAR
# =========================
def clamp(val, low, high):
    return max(low, min(val, high))


def crop_patch(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = clamp(int(x1), 0, w - 1)
    y1 = clamp(int(y1), 0, h - 1)
    x2 = clamp(int(x2), 1, w)
    y2 = clamp(int(y2), 1, h)

    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2].copy()


def get_corner_patches(img, box, patch_ratio=0.20):
    """
    box: (x1, y1, x2, y2)
    4 köşe patch döndürür.
    """
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1

    pw = max(4, int(bw * patch_ratio))
    ph = max(4, int(bh * patch_ratio))

    patches = []

    # sol üst
    patches.append(crop_patch(img, x1, y1, x1 + pw, y1 + ph))
    # sağ üst
    patches.append(crop_patch(img, x2 - pw, y1, x2, y1 + ph))
    # sol alt
    patches.append(crop_patch(img, x1, y2 - ph, x1 + pw, y2))
    # sağ alt
    patches.append(crop_patch(img, x2 - pw, y2 - ph, x2, y2))

    return [p for p in patches if p is not None and p.size > 0]


def mask_orange_pixels_bgr(patch):
    """
    Turuncu balonu kaba şekilde maskelemek için HSV tabanlı maske.
    Turuncu bölgeler 0, diğerleri 1 olarak döner.
    """
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    # Bu aralıklar senin görsellerine göre ince ayar ister.
    lower_orange = np.array([5, 80, 80], dtype=np.uint8)
    upper_orange = np.array([25, 255, 255], dtype=np.uint8)

    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    keep_mask = (orange_mask == 0).astype(np.uint8)
    return keep_mask


def normalized_rgb_signature(patch, ignore_orange=True):
    """
    Patch için normalized RGB ortalaması üretir:
        r = R/(R+G+B), g = G/(R+G+B), b = B/(R+G+B)
    """
    if patch is None or patch.size == 0:
        return None

    patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32)

    if ignore_orange:
        keep_mask = mask_orange_pixels_bgr(patch)
        valid = keep_mask > 0
    else:
        valid = np.ones(patch_rgb.shape[:2], dtype=bool)

    pixels = patch_rgb[valid]
    if len(pixels) < MIN_VALID_PIXELS:
        return None

    sums = pixels.sum(axis=1) + 1e-6
    norm = pixels / sums[:, None]

    mean_sig = norm.mean(axis=0)  # [r,g,b]
    return mean_sig


def bbox_color_signature(img, box, patch_ratio=0.20, ignore_orange=True):
    """
    4 köşe patch'in ortalamasını alıp nesnenin renk imzasını çıkarır.
    """
    patches = get_corner_patches(img, box, patch_ratio)
    sigs = []

    for p in patches:
        sig = normalized_rgb_signature(p, ignore_orange=ignore_orange)
        if sig is not None:
            sigs.append(sig)

    if len(sigs) == 0:
        return None

    sigs = np.array(sigs, dtype=np.float32)
    return sigs.mean(axis=0)  # [r,g,b]


def enemy_score(signature):
    """
    Kırmızı - mavi farkı.
    Pozitifse kırmızı baskın, negatifse mavi baskın.
    """
    r, g, b = signature
    return float(r - b)


def classify_signature(signature):
    """
    Tek başına kaba sınıflandırma.
    """
    if signature is None:
        return "unknown"

    score = enemy_score(signature)

    if score > RED_MARGIN:
        return "enemy"
    elif score < -BLUE_MARGIN:
        return "friend"
    return "uncertain"


def outlier_index(signatures):
    """
    Basit outlier seçimi:
    grubun ortalamasına en uzak imzayı döndürür.
    """
    arr = np.array(signatures, dtype=np.float32)
    center = arr.mean(axis=0)
    dists = np.linalg.norm(arr - center, axis=1)
    return int(np.argmax(dists))


def draw_box(img, box, text, color, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    t = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, t)
    ty = max(20, y1 - 8)

    cv2.rectangle(img, (x1, ty - th - 8), (x1 + tw + 8, ty), color, -1)
    cv2.putText(img, text, (x1 + 4, ty - 4), font, scale, (255, 255, 255), t, cv2.LINE_AA)


# =========================
# ANA İŞLEM
# =========================
def run_detection_and_grouping():
    model = YOLO(MODEL_PATH)

    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.*")))
    image_paths = [p for p in image_paths if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]

    if len(image_paths) == 0:
        print("Görsel bulunamadı.")
        return

    all_dets = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        results = model.predict(
            source=img,
            conf=CONF_THRES,
            iou=IOU_THRES,
            verbose=False
        )

        if len(results) == 0:
            continue

        r = results[0]

        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for b in boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            cls_id = int(b.cls[0].cpu().numpy())
            conf = float(b.conf[0].cpu().numpy())

            x1, y1, x2, y2 = map(int, xyxy)
            sig = bbox_color_signature(
                img,
                (x1, y1, x2, y2),
                patch_ratio=PATCH_RATIO,
                ignore_orange=IGNORE_ORANGE
            )

            det = {
                "img_path": img_path,
                "class_id": cls_id,
                "class_name": model.names[cls_id] if hasattr(model, "names") else str(cls_id),
                "conf": conf,
                "box": (x1, y1, x2, y2),
                "signature": sig,
                "color_score": enemy_score(sig) if sig is not None else None,
                "label": "unknown"
            }
            all_dets.append(det)

    # -------------------------
    # Aynı class içindekileri grupla
    # -------------------------
    grouped = {}
    for det in all_dets:
        grouped.setdefault(det["class_id"], []).append(det)

    for class_id, items in grouped.items():
        valid_items = [it for it in items if it["signature"] is not None]

        if len(valid_items) == 0:
            continue

        # Önce bireysel renk skoruna göre kaba etiketle
        for it in valid_items:
            it["label"] = classify_signature(it["signature"])

        # İstersen outlier mantığını da ekle
        if USE_OUTLIER_LOGIC and len(valid_items) >= 3:
            sigs = [it["signature"] for it in valid_items]
            oi = outlier_index(sigs)
            outlier_item = valid_items[oi]

            # Outlier aynı zamanda kırmızı baskınsa enemy yap
            if outlier_item["color_score"] is not None and outlier_item["color_score"] > RED_MARGIN:
                outlier_item["label"] = "enemy"

        # Son güvence:
        # aynı class içinde en yüksek kırmızı-mavi skoru taşıyanı enemy kabul et
        # ama sadece gerçekten kırmızı baskınsa
        best_red_item = max(
            valid_items,
            key=lambda x: x["color_score"] if x["color_score"] is not None else -999
        )
        if best_red_item["color_score"] is not None and best_red_item["color_score"] > RED_MARGIN:
            best_red_item["label"] = "enemy"

        # düşük kalanlar friend/uncertain olarak kalır

    return all_dets


def render_results(all_dets):
    by_image = {}
    for det in all_dets:
        by_image.setdefault(det["img_path"], []).append(det)

    saved_paths = []

    for img_path, dets in by_image.items():
        img = cv2.imread(img_path)
        if img is None:
            continue

        for det in dets:
            label = det["label"]
            sig = det["signature"]
            score = det["color_score"]

            if label == "enemy":
                color = (0, 0, 255)      # kırmızı kutu
            elif label == "friend":
                color = (255, 0, 0)      # mavi kutu
            elif label == "uncertain":
                color = (0, 255, 255)    # sarı
            else:
                color = (180, 180, 180)  # gri

            if sig is not None:
                r, g, b = sig
                text = f'{det["class_name"]} | {label} | s={score:.3f} | r={r:.2f} b={b:.2f}'
            else:
                text = f'{det["class_name"]} | {label} | no-signature'

            draw_box(img, det["box"], text, color)

        out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
        cv2.imwrite(out_path, img)
        saved_paths.append(out_path)

    return saved_paths


def simulate_sequence(image_paths, window_name="Simulation", delay_ms=1000):
    """
    Sonuç görsellerini sırayla gösterir.
    q ile çıkılır.
    """
    for p in image_paths:
        frame = cv2.imread(p)
        if frame is None:
            continue

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


def main():
    all_dets = run_detection_and_grouping()

    if not all_dets:
        print("Hiç detection bulunamadı.")
        return

    print("\n=== Detection Özeti ===")
    for det in all_dets:
        print({
            "image": os.path.basename(det["img_path"]),
            "class": det["class_name"],
            "box": det["box"],
            "score": det["color_score"],
            "label": det["label"]
        })

    rendered = render_results(all_dets)

    print("\nKaydedilen sonuçlar:")
    for p in rendered:
        print(p)

    # Simülasyon gibi sırayla oynat
    simulate_sequence(rendered, delay_ms=1200)


if __name__ == "__main__":
    main()