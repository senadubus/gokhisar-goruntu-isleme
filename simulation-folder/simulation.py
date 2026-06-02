import cv2
import numpy as np
import math
import time
from ultralytics import YOLO


# =========================================================
# 1) MODELİ YÜKLE
# =========================================================
MODEL_PATH = "best.pt"   # kendi .pt dosyan
model = YOLO(MODEL_PATH)


# =========================================================
# 2) HEDEF GÖRSELLERİNİ YÜKLE
#    Bunları sen vereceksin
# =========================================================
IMAGE_PATHS = [
    "images2/target1.png",
    "images2/target2.png",
    "images2/target3.png",
    "images2/target4.png",
]

images = []
for p in IMAGE_PATHS:
    img = cv2.imread(p, cv2.IMREAD_UNCHANGED)  # alpha varsa da al
    if img is None:
        raise FileNotFoundError(f"Görsel yüklenemedi: {p}")
    images.append(img)


# =========================================================
# 3) AYARLAR
# =========================================================
SCREEN_W, SCREEN_H = 1280, 720
CENTER_X, CENTER_Y = SCREEN_W // 2, SCREEN_H // 2

FPS = 30
SIM_DURATION = 12.0  # bir yaklaşma döngüsü süresi
LANES = [180, 460, 780, 1080]

# YOLO inference size
YOLO_IMGSZ = 640
YOLO_CONF = 0.25

# Turuncu ROI merkezi için HSV aralığı
LOWER_ORANGE = np.array([5, 120, 120], dtype=np.uint8)
UPPER_ORANGE = np.array([25, 255, 255], dtype=np.uint8)
KERNEL = np.ones((5, 5), np.uint8)

MIN_CONTOUR_AREA = 20
MIN_CIRCULARITY = 0.50
MIN_RADIUS = 2


# =========================================================
# 4) YARDIMCI FONKSİYONLAR
# =========================================================
def split_foreground_and_mask(img):
    """
    PNG alpha varsa alpha'yı maske olarak kullanır.
    Yoksa siyah arka planı maske dışı kabul eder.
    """
    if len(img.shape) == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        return bgr, alpha

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    return img, mask


def overlay_safe(background, overlay_bgr, mask, x, y):
    """
    Görsel taşsa da hata vermeden basar.
    """
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
    """
    ROI crop -> HSV mask -> morphology -> largest contour
    -> circularity -> minEnclosingCircle -> global center
    """
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
    """
    Merkeze en yakın bbox merkezini ana hedef seç.
    detections: [{"bbox":(...), ...}, ...]
    """
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
# 5) SİMÜLASYON DÖNGÜSÜ
# =========================================================
frame_count = 0

print("Simülasyon başladı. Çıkmak için q bas.")

while True:
    loop_start = time.time()

    # Siyah arka plan
    frame = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    draw_crosshair(frame)

    # İlerleme: 0 -> 1
    t = frame_count / FPS
    progress = (t % SIM_DURATION) / SIM_DURATION

    # Uzakta küçük, yakında büyük
    scale = 0.08 + (progress ** 2) * 1.7

    # Hareketli hedefleri frame'e yerleştir
    for i, img in enumerate(images):
        base_h, base_w = img.shape[:2]

        new_w = int(base_w * scale)
        new_h = int(base_h * scale)

        if new_w < 8 or new_h < 8:
            continue

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        fg, mask = split_foreground_and_mask(resized)

        # Hafif slalom
        x_offset = 80 * math.sin(t * (1.3 + i * 0.25))
        y_offset = 45 * math.cos(t * (1.0 + i * 0.15) + i)

        x = int(LANES[i] + x_offset - new_w / 2)
        y = int(CENTER_Y + y_offset - new_h / 2)

        overlay_safe(frame, fg, mask, x, y)

    # -----------------------------------------------------
    # 6) YOLO NESNE TESPİTİ
    # -----------------------------------------------------
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

                # ROI içinde turuncu dairesel merkez bul
                orange_info = detect_orange_center_in_bbox(frame, (x1, y1, x2, y2))

                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "cls_id": cls_id,
                    "class_name": class_name,
                    "conf": conf,
                    "orange_info": orange_info,
                })

    # Ana hedef seç
    main_idx = choose_main_target(detections)

    # -----------------------------------------------------
    # 7) ÇİZİM
    # -----------------------------------------------------
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        name = det["class_name"]
        conf = det["conf"]
        orange_info = det["orange_info"]

        if i == main_idx:
            box_color = (0, 0, 255)   # kırmızı
            txt_color = (0, 255, 0)
        else:
            box_color = (0, 255, 255) # sarı
            txt_color = (255, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            frame,
            f"{name} {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            txt_color,
            2
        )

        # bbox merkezi
        bx = (x1 + x2) // 2
        by = (y1 + y2) // 2
        cv2.circle(frame, (bx, by), 4, (255, 255, 0), -1)

        # Turuncu merkez bulunduysa çiz
        if orange_info is not None:
            gx, gy = orange_info["global_center"]
            radius = int(orange_info["radius"])
            circularity = orange_info["circularity"]

            cv2.circle(frame, (gx, gy), max(3, radius), (0, 255, 0), 2)
            cv2.circle(frame, (gx, gy), 3, (0, 255, 0), -1)
            cv2.putText(
                frame,
                f"C={circularity:.2f}",
                (gx + 8, gy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
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
                0.6,
                (0, 255, 0),
                2
            )

    # Bilgi yazıları
    sim_dist = round(15.0 * (1.0 - progress), 2)
    cv2.putText(
        frame,
        f"Simule Mesafe: {sim_dist} m",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )
    cv2.putText(
        frame,
        f"Detection Count: {len(detections)}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("YOLO Simulation", frame)

    elapsed = time.time() - loop_start
    wait_ms = max(1, int((1.0 / FPS - elapsed) * 1000))

    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord("q"):
        break

    frame_count += 1

cv2.destroyAllWindows()