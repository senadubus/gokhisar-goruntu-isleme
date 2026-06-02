import cv2
import numpy as np
import math
import time
from ultralytics import YOLO


# =========================================================
# 1) MODELİ YÜKLE
# =========================================================
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)


# =========================================================
# 2) HEDEF GÖRSELLERİNİ YÜKLE
# =========================================================
IMAGE_PATHS = [
    "images/1.png",
    "images/5.png",
    "images/5.png",
]

images = []
for p in IMAGE_PATHS:
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Görsel yüklenemedi: {p}")
    images.append(img)


# =========================================================
# 3) AYARLAR
# =========================================================
SCREEN_W, SCREEN_H = 1280, 720
CENTER_X, CENTER_Y = SCREEN_W // 2, SCREEN_H // 2

FPS = 30
SIM_DURATION = 12.0
LANES = [180, 460, 780, 1080]

YOLO_IMGSZ = 640
YOLO_CONF = 0.25

# Turuncu ROI (balon tespiti için)
LOWER_ORANGE = np.array([5, 120, 120], dtype=np.uint8)
UPPER_ORANGE = np.array([25, 255, 255], dtype=np.uint8)
KERNEL = np.ones((5, 5), np.uint8)

MIN_CONTOUR_AREA = 20
MIN_CIRCULARITY = 0.50
MIN_RADIUS = 2

# Renk imzası — patch boyut oranı
PATCH_RATIO = 0.20

# Outlier tespiti için minimum mesafe eşiği
# (bu kadar küçük fark gürültüdür, yok say)
OUTLIER_MIN_DIFF = 0.02


# =========================================================
# 4) YARDIMCI FONKSİYONLAR (orijinal)
# =========================================================
def split_foreground_and_mask(img):
    if len(img.shape) == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        return bgr, alpha
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    return img, mask


def overlay_safe(background, overlay_bgr, mask, x, y):
    h, w = overlay_bgr.shape[:2]
    y1, y2 = max(0, y), min(background.shape[0], y + h)
    x1, x2 = max(0, x), min(background.shape[1], x + w)
    oy1, oy2 = max(0, -y), min(h, background.shape[0] - y)
    ox1, ox2 = max(0, -x), min(w, background.shape[1] - x)
    if y1 >= y2 or x1 >= x2:
        return
    roi = background[y1:y2, x1:x2]
    overlay_crop = overlay_bgr[oy1:oy2, ox1:ox2]
    mask_crop = mask[oy1:oy2, ox1:ox2]
    mask_inv = cv2.bitwise_not(mask_crop)
    bg_part = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg_part = cv2.bitwise_and(overlay_crop, overlay_crop, mask=mask_crop)
    background[y1:y2, x1:x2] = cv2.add(bg_part, fg_part)


def detect_orange_center_in_bbox(frame, bbox):
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(frame.shape[1], int(x2))
    y2 = min(frame.shape[0], int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_ORANGE, UPPER_ORANGE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < MIN_CONTOUR_AREA:
        return None
    perimeter = cv2.arcLength(largest, True)
    if perimeter == 0:
        return None
    circularity = 4 * math.pi * area / (perimeter * perimeter)
    if circularity < MIN_CIRCULARITY:
        return None
    (cx_roi, cy_roi), radius = cv2.minEnclosingCircle(largest)
    if radius < MIN_RADIUS:
        return None
    gx = int(cx_roi + x1)
    gy = int(cy_roi + y1)
    return {
        "global_center": (gx, gy),
        "roi_center": (int(cx_roi), int(cy_roi)),
        "radius": float(radius),
        "circularity": float(circularity),
        "area": float(area),
    }


def draw_crosshair(frame):
    cv2.line(frame, (CENTER_X - 20, CENTER_Y), (CENTER_X + 20, CENTER_Y), (255, 0, 0), 2)
    cv2.line(frame, (CENTER_X, CENTER_Y - 20), (CENTER_X, CENTER_Y + 20), (255, 0, 0), 2)


def choose_main_target(detections):
    if not detections:
        return None
    best_idx = None
    best_dist = float("inf")
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        dist = math.hypot(cx - CENTER_X, cy - CENTER_Y)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


# =========================================================
# 5) RENKİMZASI FONKSİYONLARI (YENİ)
# =========================================================

def get_color_signature(frame, bbox, orange_mask_hsv=True):
    """
    Bbox'un 4 köşesinden patch alır, turuncu piksellerı opsiyonel olarak
    ignore eder, normalized RGB ortalaması döner: (r, g, b) her biri [0,1].

    Parametreler
    ------------
    frame           : BGR frame (SCREEN_H x SCREEN_W x 3)
    bbox            : (x1, y1, x2, y2) int
    orange_mask_hsv : True ise turuncu pikseleri (balon) ignore et

    Dönen değer
    -----------
    np.ndarray şeklinde [r_norm, g_norm, b_norm] veya None (hesaplanamadıysa)
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Bbox içinde kalacak şekilde sınırla
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    bw = x2 - x1
    bh = y2 - y1

    if bw < 10 or bh < 10:
        return None

    pw = max(4, int(bw * PATCH_RATIO))
    ph = max(4, int(bh * PATCH_RATIO))

    # 4 köşe: (sol_üst, sağ_üst, sol_alt, sağ_alt)
    patches = [
        frame[y1       : y1 + ph,  x1       : x1 + pw],   # sol üst
        frame[y1       : y1 + ph,  x2 - pw  : x2      ],   # sağ üst
        frame[y2 - ph  : y2,       x1       : x1 + pw],    # sol alt
        frame[y2 - ph  : y2,       x2 - pw  : x2      ],   # sağ alt
    ]

    sig_accumulator = np.zeros(3, dtype=np.float64)  # (r, g, b) normalized toplam
    valid_patch_count = 0

    for patch in patches:
        if patch.size == 0:
            continue

        # Turuncu maskeleme (opsiyonel)
        if orange_mask_hsv:
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            orange_m = cv2.inRange(hsv_patch, LOWER_ORANGE, UPPER_ORANGE)
            # Non-orange pikselleri seç
            valid_pixels = patch[orange_m == 0]
        else:
            valid_pixels = patch.reshape(-1, 3)

        if len(valid_pixels) < 5:
            continue

        # BGR → float
        B = valid_pixels[:, 0].astype(np.float64)
        G = valid_pixels[:, 1].astype(np.float64)
        R = valid_pixels[:, 2].astype(np.float64)

        total = R + G + B
        # Siyah pikselleri (total ≈ 0) atla
        nonzero = total > 10
        if nonzero.sum() < 3:
            continue

        r_norm = (R[nonzero] / total[nonzero]).mean()
        g_norm = (G[nonzero] / total[nonzero]).mean()
        b_norm = (B[nonzero] / total[nonzero]).mean()

        sig_accumulator += np.array([r_norm, g_norm, b_norm])
        valid_patch_count += 1

    if valid_patch_count == 0:
        return None

    return sig_accumulator / valid_patch_count  # 4 patch ortalaması


def find_color_outlier(detections):
    """
    Aynı class_id'ye sahip tespitler arasında renk imzası outlier'ını bulur.

    Algoritma:
      - Her class için tespitleri grupla.
      - Grupta 3+ nesne varsa:
          * Her nesnenin imzası ile gruptaki diğerlerinin ortalaması
            arasındaki Öklid mesafesini hesapla.
          * En uzak olanı outlier işaretle.
      - Grupta 2 nesne varsa: hangi class'taki mesafe daha büyükse o çift içinde
        farklı olan outlier (ikisi de aynı derecede "uzak" olduğundan ikisini de işaretle).

    Her detection dict'ine "color_sig", "is_color_outlier" anahtarları eklenir.
    """
    # Önce imzaları hesapla ve dict'e yaz
    # (frame erişimi burada yok, imzalar dışarıda hesaplanıp geçilmeli;
    #  bu fonksiyon sadece "color_sig" zaten dolu olan detections üzerinde çalışır)

    # Class bazlı gruplama
    from collections import defaultdict
    class_groups = defaultdict(list)  # cls_id -> [idx, ...]

    for i, det in enumerate(detections):
        det.setdefault("is_color_outlier", False)
        if det.get("color_sig") is not None:
            class_groups[det["cls_id"]].append(i)

    for cls_id, indices in class_groups.items():
        if len(indices) < 2:
            continue  # tek nesne, karşılaştırma yok

        sigs = np.array([detections[i]["color_sig"] for i in indices])  # (N, 3)

        if len(indices) == 2:
            # Sadece 2 nesne: farkı hesapla, eşiğin üstündeyse her ikisini de değil,
            # hiçbirini outlier yapma (hangisi farklı bilinmez); sadece fark bilgisini kaydet.
            diff = np.linalg.norm(sigs[0] - sigs[1])
            if diff > OUTLIER_MIN_DIFF:
                # Hangisi "doğal" renk bilinmiyor; sadece bilgi amaçlı işaretle
                # (isteğe bağlı davranış)
                pass
            continue

        # 3+ nesne: her nesne için diğerlerinin ortalamasına uzaklık
        distances = []
        for i_local, idx in enumerate(indices):
            others = np.delete(sigs, i_local, axis=0)
            mean_others = others.mean(axis=0)
            dist = np.linalg.norm(sigs[i_local] - mean_others)
            distances.append(dist)

        distances = np.array(distances)
        max_dist_local = np.argmax(distances)
        max_dist_val = distances[max_dist_local]

        if max_dist_val > OUTLIER_MIN_DIFF:
            outlier_global_idx = indices[max_dist_local]
            detections[outlier_global_idx]["is_color_outlier"] = True

    return detections


def draw_color_patch_debug(frame, bbox, patch_ratio=PATCH_RATIO):
    """
    Debug: bbox üzerinde 4 patch bölgesini küçük dikdörtgen ile gösterir.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
    bw = x2 - x1; bh = y2 - y1
    pw = max(4, int(bw * patch_ratio))
    ph = max(4, int(bh * patch_ratio))
    color = (180, 180, 0)
    cv2.rectangle(frame, (x1, y1), (x1 + pw, y1 + ph), color, 1)
    cv2.rectangle(frame, (x2 - pw, y1), (x2, y1 + ph), color, 1)
    cv2.rectangle(frame, (x1, y2 - ph), (x1 + pw, y2), color, 1)
    cv2.rectangle(frame, (x2 - pw, y2 - ph), (x2, y2), color, 1)


# =========================================================
# 6) SİMÜLASYON DÖNGÜSÜ
# =========================================================
frame_count = 0
print("Simülasyon başladı. Çıkmak için q bas.")

while True:
    loop_start = time.time()

    frame = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    draw_crosshair(frame)

    t = frame_count / FPS
    progress = (t % SIM_DURATION) / SIM_DURATION
    scale = 0.08 + (progress ** 2) * 1.7

    for i, img in enumerate(images):
        base_h, base_w = img.shape[:2]
        new_w = int(base_w * scale)
        new_h = int(base_h * scale)
        if new_w < 8 or new_h < 8:
            continue
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        fg, mask = split_foreground_and_mask(resized)
        x_offset = 80 * math.sin(t * (1.3 + i * 0.25))
        y_offset = 45 * math.cos(t * (1.0 + i * 0.15) + i)
        x = int(LANES[i] + x_offset - new_w / 2)
        y = int(CENTER_Y + y_offset - new_h / 2)
        overlay_safe(frame, fg, mask, x, y)

    # ----------------------------------------------------------
    # 7) YOLO TESPİTİ
    # ----------------------------------------------------------
    results = model.predict(
        source=frame,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        verbose=False
    )

    detections = []

    if len(results) > 0:
        r = results[0]
        if r.boxes is not None:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy.tolist()
                cls_id = int(box.cls[0].item()) if box.cls is not None else -1
                conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                class_name = model.names.get(cls_id, str(cls_id))

                orange_info = detect_orange_center_in_bbox(frame, (x1, y1, x2, y2))

                # --- YENİ: renk imzasını hesapla ---
                color_sig = get_color_signature(frame, (x1, y1, x2, y2),
                                                orange_mask_hsv=True)

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "cls_id": cls_id,
                    "class_name": class_name,
                    "conf": conf,
                    "orange_info": orange_info,
                    "color_sig": color_sig,
                    "is_color_outlier": False,   # find_color_outlier dolduracak
                })

    # --- YENİ: outlier tespiti ---
    detections = find_color_outlier(detections)

    main_idx = choose_main_target(detections)

    # ----------------------------------------------------------
    # 8) ÇİZİM
    # ----------------------------------------------------------
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        name = det["class_name"]
        conf = det["conf"]
        orange_info = det["orange_info"]
        is_outlier = det["is_color_outlier"]
        color_sig = det["color_sig"]

        if i == main_idx:
            box_color = (0, 0, 255)     # kırmızı — ana hedef
            txt_color = (0, 255, 0)
        else:
            box_color = (0, 255, 255)   # sarı
            txt_color = (255, 255, 255)

        # Outlier ise bbox çerçevesini magenta yap
        if is_outlier:
            box_color = (255, 0, 255)   # magenta — farklı renk!

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            frame,
            f"{name} {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, txt_color, 2
        )

        # --- YENİ: renk imzasını ekrana yaz ---
        if color_sig is not None:
            r_n, g_n, b_n = color_sig
            sig_text = f"r={r_n:.2f} g={g_n:.2f} b={b_n:.2f}"
            cv2.putText(
                frame, sig_text,
                (x1, max(40, y1 - 28)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1
            )

        # --- YENİ: outlier etiketi ---
        if is_outlier:
            cv2.putText(
                frame, "FARKLI RENK",
                (x1, min(SCREEN_H - 10, y2 + 48)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 0, 255), 2
            )

        # --- YENİ: debug — 4 patch köşelerini göster (isteğe bağlı) ---
        draw_color_patch_debug(frame, (x1, y1, x2, y2))

        # Bbox merkezi
        bx = (x1 + x2) // 2
        by = (y1 + y2) // 2
        cv2.circle(frame, (bx, by), 4, (255, 255, 0), -1)

        # Turuncu merkez
        if orange_info is not None:
            gx, gy = orange_info["global_center"]
            radius = int(orange_info["radius"])
            circularity = orange_info["circularity"]
            cv2.circle(frame, (gx, gy), max(3, radius), (0, 255, 0), 2)
            cv2.circle(frame, (gx, gy), 3, (0, 255, 0), -1)
            cv2.putText(
                frame, f"C={circularity:.2f}",
                (gx + 8, gy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1
            )

        # Ana hedefe lock çizgisi
        if i == main_idx:
            target_x, target_y = bx, by
            if orange_info is not None:
                target_x, target_y = orange_info["global_center"]
            cv2.line(frame, (CENTER_X, CENTER_Y), (target_x, target_y), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"LOCKED | ERR X:{target_x - CENTER_X} Y:{target_y - CENTER_Y}",
                (max(10, x1), min(SCREEN_H - 10, y2 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2
            )

    # Bilgi yazıları
    sim_dist = round(15.0 * (1.0 - progress), 2)
    cv2.putText(frame, f"Simule Mesafe: {sim_dist} m",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Detection Count: {len(detections)}",
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Outlier sayısını da yaz
    outlier_count = sum(1 for d in detections if d["is_color_outlier"])
    cv2.putText(frame, f"Color Outlier: {outlier_count}",
                (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

    cv2.imshow("YOLO Simulation", frame)

    elapsed = time.time() - loop_start
    wait_ms = max(1, int((1.0 / FPS - elapsed) * 1000))

    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord("q"):
        break

    frame_count += 1

cv2.destroyAllWindows()