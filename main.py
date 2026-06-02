import time
from constants import StateMachine, TargetState, TOLERANS, BEKLEME_SURESI, KAMERA_MERKEZ_X, KAMERA_MERKEZ_Y
from models import TargetPair
import helpers

def main():
    ASAMA_MODU = 3  # Arayüzden gelen mod bilgisi (2 veya 3)
    
    system_state = StateMachine.SCANNING
    target_pool = []
    active_target = None
    
    while True:
        frame, fps = "Kamera_Görüntüsü", 30  # Mock kamera akışı
        
        # --- ACİL DURUM ---
        if frame is None or fps < 5:
            system_state = StateMachine.EMERGENCY_STOP
            helpers.send_to_raspberry(0, 0, 0, lock_status=0)
            continue
            
        # YOLO Çıktıları (Mock)
        ham_maketler = [{"id": 1, "sinif": 0, "bbox": (100, 50, 150, 100)}]
        tum_balonlar = [{"merkez": (125, 120)}]
        
        # --- 1. HAVUZU VE EŞLEŞMELERİ GÜNCELLE ---
        bu_karedeki_ciftler = []
        for maket in ham_maketler:
            bagli_balon = helpers.check_balloon_in_extended_roi(maket["bbox"], tum_balonlar)
            if bagli_balon:
                cift = TargetPair(maket["id"], maket["sinif"], maket["bbox"], bagli_balon)
                
                if ASAMA_MODU == 2:
                    cift.state = TargetState.SINIFLANDIRMA
                    bu_karedeki_ciftler.append(cift)
                elif ASAMA_MODU == 3:
                    if helpers.analyze_color_iff(maket["bbox"]) == "KIRMIZI":
                        cift.state = TargetState.SINIFLANDIRMA
                        bu_karedeki_ciftler.append(cift)
        
        target_pool = bu_karedeki_ciftler
        
        # --- 2. HEDEF SADAKATİ (STICKINESS) & SEÇİM ---
        if active_target is not None and active_target.state != TargetState.IMHA:
            active_target = next((x for x in target_pool if x.id == active_target.id), None)
        else:
            active_target = helpers.select_best_target(target_pool)
            if active_target: system_state = StateMachine.DETECT

        # --- 3. DURUM MAKİNESİ EYLEMLERİ ---
        if active_target is None:
            system_state = StateMachine.SCANNING
            helpers.send_to_raspberry(0, 0, 0, lock_status=0)
        else:
            # Balon merkezli hata hesaplama
            bx, by = active_target.balon_bbox["merkez"]
            active_target.error_x = bx - KAMERA_MERKEZ_X
            active_target.error_y = by - KAMERA_MERKEZ_Y
            
            if active_target.state == TargetState.SINIFLANDIRMA:
                active_target.state = TargetState.TAKIP
                system_state = StateMachine.ALARM
                helpers.send_to_raspberry(active_target.error_x, active_target.error_y, active_target.sinif_id, 0)
                
            elif active_target.state == TargetState.TAKIP:
                system_state = StateMachine.TRACK
                if abs(active_target.error_x) < TOLERANS and abs(active_target.error_y) < TOLERANS:
                    active_target.state = TargetState.DEGERLENDIRME
                    active_target.seen_counter = 0
                helpers.send_to_raspberry(active_target.error_x, active_target.error_y, active_target.sinif_id, 0)
                
            elif active_target.state == TargetState.DEGERLENDIRME:
                system_state = StateMachine.DEGERLENDIRME
                active_target.seen_counter += 1
                if abs(active_target.error_x) < TOLERANS and abs(active_target.error_y) < TOLERANS:
                    if active_target.seen_counter >= BEKLEME_SURESI:
                        active_target.state = TargetState.CONTROL
                else:
                    active_target.state = TargetState.TAKIP
                    system_state = StateMachine.TRACK
                helpers.send_to_raspberry(active_target.error_x, active_target.error_y, active_target.sinif_id, 0)
                
            elif active_target.state == TargetState.CONTROL:
                system_state = StateMachine.KILL
                # PC kilitlendi, yetki devri yapılıyor (Krokodil hatası yok, ates_et ezilmiyor!)
                helpers.send_to_raspberry(active_target.error_x, active_target.error_y, active_target.sinif_id, lock_status=1)
                
        time.sleep(0.033)  # ~30 FPS Döngü Sabitleyici

if __name__ == "__main__":
    main()