import sys
import time
from modules.dry_eeg_support import DryEEGDevice
from modules.realtime_processing import RealTimeEngine
from modules.zero_calibration import ZeroCalibrationBCI
from modules.rehab_training import NeuroRehabTrainer


def clear_screen():
    print("\n" * 2)


def main_dashboard():
    print("================================================================")
    print("   🧠 INTEGRATED BCI COMMERCIAL SUITE (v1.0)")
    print("   Supported by Health & Medical Technology R&D Program")
    print("================================================================")
    print("1. [Hardware] Connect Dry-EEG Headset")
    print("2. [Software] Start Zero-Calibration Mode (Plug & Play)")
    print("3. [Therapy]  Launch Neuro-Rehabilitation Training")
    print("4. [Engine]   View Real-Time Signal Monitor")
    print("5. Exit")

    choice = input("\n👉 Select Module: ").strip()

    if choice == "1":
        device = DryEEGDevice()
        if device.connect():
            device.check_impedance()

    elif choice == "2":
        app = ZeroCalibrationBCI()
        app.initialize_session()
        app.run_demo()

    elif choice == "3":
        trainer = NeuroRehabTrainer(target_emotion="Calmness")
        trainer.start_session()

    elif choice == "4":
        print("\n[MONITOR] Initializing Real-Time Engine...")
        engine = RealTimeEngine()
        print("[MONITOR] Engine Active. Listening to stream...")
        # Demo loop
        for i in range(5):
            time.sleep(0.5)
            # Simulate raw data coming in
            packet = [10.2, -4.5, 33.1, 2.0, -15.5, 0.0, 1.2, 5.5]
            clean = engine.process_packet(packet)
            print(f"   Processing Packet {i + 1}: Filtered & Artifacts Removed.")

    elif choice == "5":
        print("\nExiting System. Goodbye!")
        sys.exit()

    else:
        print("Invalid Selection.")

    input("\n[PRESS ENTER TO RETURN TO MENU]")
    clear_screen()
    main_dashboard()


if __name__ == "__main__":
    main_dashboard()