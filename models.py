from constants import TargetState

class TargetPair:
    def __init__(self, target_id, sinif_id, maket_bbox, balon_bbox):
        self.id = target_id
        self.sinif_id = sinif_id     # 0: IHA, 1: F16 vb.
        self.maket_bbox = maket_bbox # (x_ust, y_ust, x_alt, y_alt)
        self.balon_bbox = balon_bbox
        self.state = TargetState.TESPIT
        self.seen_counter = 0
        self.lost_counter = 0
        self.error_x = 0
        self.error_y = 0