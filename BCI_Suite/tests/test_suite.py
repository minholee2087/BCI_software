import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.realtime_processing import RealTimeEngine

class TestBCIEngine(unittest.TestCase):

    def setUp(self):
        """Initialize the engine before each test."""
        self.engine = RealTimeEngine(sampling_rate=250)

    def test_buffer_integrity(self):
        """Test if data buffering works correctly."""
        dummy_packet = [10.0] * 30  # 30 channels

        # Send 10 packets
        for _ in range(10):
            self.engine.process_packet(dummy_packet)

        # Check if buffer is not empty
        self.assertTrue(len(self.engine.buffer) > 0, "Buffer should not be empty")

    def test_artifact_removal(self):
        """Test if high-amplitude artifacts are clipped."""
        # Create a spike (value 500 is way above normal EEG)
        spike_packet = [500.0, -500.0, 0.0]
        cleaned = self.engine.process_packet(spike_packet)

        # Check if values were clipped to +/- 80 (as defined in our code)
        self.assertTrue(cleaned[0] <= 80, "Positive artifact was not clipped")
        self.assertTrue(cleaned[1] >= -80, "Negative artifact was not clipped")


if __name__ == '__main__':
    unittest.main()