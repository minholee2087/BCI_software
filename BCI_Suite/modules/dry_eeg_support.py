import time
import random
import threading

class DryEEGDevice:
    """
    [Deliverable 1.2.1] Practical Dry-EEG Device Support Software
    - Handles Bluetooth/Serial connections.
    - Monitors electrode impedance (contact quality).
    - Streams raw EEG packets.
    """

    def __init__(self, device_name="NeuroLink-Pro X1"):
        self.device_name = device_name
        self.is_connected = False
        self.is_streaming = False
        self.battery_level = 100

    def connect(self):
        print(f"\n[HARDWARE] Scanning for {self.device_name}...")
        time.sleep(1.5)  # Simulate bluetooth handshake
        self.is_connected = True
        print(f"[HARDWARE] Connected to {self.device_name} (ID: AA:BB:CC:11:22)")
        return True

    def check_impedance(self):
        """Simulate checking if the dry electrodes are touching skin properly."""
        if not self.is_connected:
            print("[ERROR] Device not connected.")
            return False

        print("[HARDWARE] Measuring Impedance...")
        time.sleep(1.0)
        # Simulate impedance values for 8 channels (in kOhms)
        # Dry EEG usually requires < 50kOhm, Wet < 10kOhm
        impedances = [random.randint(5, 40) for _ in range(8)]
        status = "Good" if max(impedances) < 50 else "Poor"

        print(f"[HARDWARE] Channel Status: {impedances} kΩ -> Signal Quality: {status}")
        return status == "Good"

    def stream(self, duration=5):
        """Simulate data streaming for a fixed duration."""
        if not self.is_connected: return
        print(f"[HARDWARE] Streaming Raw Data ({duration}s)...")
        self.is_streaming = True

        # Simulate a stream loop
        start = time.time()
        while time.time() - start < duration:
            # Generate dummy EEG packet (8 channels)
            packet = [round(random.uniform(-100, 100), 2) for _ in range(8)]
            # In a real app, this would be yielded to the processor
            time.sleep(0.1)

        self.is_streaming = False
        print("[HARDWARE] Stream paused.")
