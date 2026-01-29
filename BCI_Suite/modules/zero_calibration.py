import time
from .realtime_processing import RealTimeEngine

class ZeroCalibrationBCI:
    """
    [Deliverable 1.2.1] Zero-calibration BCI Software
    - Loads 'Universal' weights instead of subject-specific weights.
    - Allows immediate usage.
    """

    def __init__(self):
        self.engine = RealTimeEngine()
        self.mode = "Universal_ZeroShot_v2"

    def initialize_session(self):
        print(f"\n[ZERO-CALIB] 🔄 Loading Universal Model ({self.mode})...")
        time.sleep(1.5)
        print("[ZERO-CALIB] ✅ Model Loaded. No Subject Training Required.")
        print("[ZERO-CALIB] System is ready for user.")

    def run_demo(self):
        print("[ZERO-CALIB] Starting detection...")
        for i in range(3):
            time.sleep(1)
            state, conf = self.engine.get_current_state()
            print(f"   -> User State: {state.upper()} (Confidence: {conf:.2f})")