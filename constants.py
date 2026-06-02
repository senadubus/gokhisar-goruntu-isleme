from enum import Enum

class StateMachine(Enum):
    IDLE = 1
    SCANNING = 2
    DETECT = 3
    ALARM = 4
    TRACK = 5
    DEGERLENDIRME = 6
    KILL = 7
    LOST = 8
    EMERGENCY_STOP = 9

class TargetState(Enum):
    TESPIT = 1
    SINIFLANDIRMA = 2
    TAKIP = 3
    DEGERLENDIRME = 4
    CONTROL = 5
    IMHA = 6

# --- EŞİK DEĞERLERİ ---
TOLERANS = 15             # Kilitlenme toleransı (piksel)
BEKLEME_SURESI = 6        # Atış onay döngü sayısı (30 FPS'te ~200 ms)
KAMERA_MERKEZ_X = 320     # 640x480 kamera varsayıldı
KAMERA_MERKEZ_Y = 240