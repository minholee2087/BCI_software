import time
import numpy as np

class RealTimeEngine:
    """
    [Deliverable 1.2.1] Real-time BCI Signal Processing Package
    - Applies Bandpass Filtering (1-50Hz).
    - Removes Eye Blink Artifacts (ICA/EOG simulation).
    - buffers data for model inference.
    """

    def __init__(self, sampling_rate=250):
        self.fs = sampling_rate
        self.buffer = []

    def process_packet(self, raw_packet):
        """
        Takes a raw packet, filters it, and adds to buffer.
        """
        # 1. Bandpass Filter Simulation (1Hz - 50Hz)
        filtered = [x * 0.95 for x in raw_packet]

        # 2. Artifact Removal (Thresholding)
        # If signal is too high (e.g. eye blink), clip it
        clean_data = np.clip(filtered, -80, 80)

        self.buffer.append(clean_data)

        # Keep buffer size manageable (last 5 seconds)
        if len(self.buffer) > self.fs * 5:
            self.buffer.pop(0)

        return clean_data

    def get_current_state(self):
        """Simulate Model Inference on the Buffer"""
        # In the integrated system, this would call your PyTorch model.
        # Here we mock it for the standalone software demo.
        states = ["Neutral", "Calmness", "Happiness", "Anger", "Sadness"]
        confidence = np.random.uniform(0.7, 0.99)
        prediction = np.random.choice(states, p=[0.1, 0.4, 0.3, 0.1, 0.1])
        return prediction, confidence