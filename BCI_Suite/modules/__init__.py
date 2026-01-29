# BCI_Suite/modules/__init__.py

from .dry_eeg_support import DryEEGDevice
from .realtime_processing import RealTimeEngine
from .rehab_training import NeuroRehabTrainer
from .zero_calibration import ZeroCalibrationBCI

__all__ = [
    'DryEEGDevice',
    'RealTimeEngine',
    'NeuroRehabTrainer',
    'ZeroCalibrationBCI'
]

print("[INIT] BCI Commercial Modules Loaded.")

__version__ = "1.0.0"
__author__ = "Minho Lee (Principal Investigator)"