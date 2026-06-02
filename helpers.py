import constants

def check_balloon_in_extended_roi(maket_bbox, tum_balonlar):
    """Maketin altını kendi boyu kadar uzatıp içinde balon arar."""
    x_ust, y_ust, x_alt, y_alt = maket_bbox
    maket_yukseklik = y_alt - y_ust
    
    # Alt taraftaki sanal arama alanı (ROI)
    roi_y_ust = y_alt
    roi_y_alt = y_alt + maket_yukseklik
    
    for balon in tum_balonlar:
        bx, by = balon["merkez"]
        if x_ust <= bx <= x_alt and roi_y_ust <= by <= roi_y_alt:
            return balon
    return None

def analyze_color_iff(maket_bbox):
    """Kutunun merkezinden dost/düşman analizi yapar."""
    return "KIRMIZI"  # Veya "MAVI" (Gerçek kodda OpenCV maskesi gelecek)

def select_best_target(target_pool):
    """Manhattan metriğiyle merkeze en yakın olanı cımbızlar."""
    en_iyi_hedef = None
    en_kucuk_skor = float('inf')
    for hedef in target_pool:
        skor = abs(hedef.error_x) + abs(hedef.error_y)
        if skor < en_kucuk_skor:
            en_kucuk_skor = skor
            en_iyi_hedef = hedef
    return en_iyi_hedef

def send_to_raspberry(error_x, error_y, sinif_id, lock_status):
    """Veriyi paketleyip seri porttan Raspberry Pi'ye atar."""
    paket = {
        "hata_x": int(error_x),
        "hata_y": int(error_y),
        "sinif_id": int(sinif_id),
        "lock_status": int(lock_status)  # 0: Takip, 1: Ateş Yetkisi!
    }
    print(f"[PC -> PI] Aktarılan Paket: {paket}")