import numpy as np
from abc import ABC, abstractmethod
import time


class EEGSource(ABC):
    @abstractmethod
    def get_sample(self, n_samples=7):
        pass
    
    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def stop(self):
        pass
    
    @property
    @abstractmethod
    def n_channels(self):
        pass
    
    @property
    @abstractmethod
    def sample_rate(self):
        pass


class SimulatedEEGSource(EEGSource):
    def __init__(self, n_channels=30, sample_rate=256):
        self._n_channels = n_channels
        self._sample_rate = sample_rate
        self.t = 0
        self.emotion_state = 'neutral'
        self.emotion_intensity = 0.5
        self._running = False
        
    @property
    def n_channels(self):
        return self._n_channels
    
    @property
    def sample_rate(self):
        return self._sample_rate
        
    def set_emotion(self, emotion, intensity=0.5):
        self.emotion_state = emotion
        self.emotion_intensity = np.clip(intensity, 0, 1)
        
    def start(self):
        self._running = True
        self.t = 0
        
    def stop(self):
        self._running = False
        
    def get_sample(self, n_samples=7):
        t = np.linspace(self.t, self.t + n_samples/self._sample_rate, n_samples)
        self.t += n_samples/self._sample_rate
        
        data = np.zeros((n_samples, self._n_channels))
        
        for ch in range(self._n_channels):
            phase_offset = ch * 0.1
            
            delta = 0.2 * np.sin(2 * np.pi * 2 * t + phase_offset)
            theta = 0.4 * np.sin(2 * np.pi * 6 * t + phase_offset * 1.5)
            alpha = 0.5 * np.sin(2 * np.pi * 10 * t + phase_offset * 2)
            beta = 0.3 * np.sin(2 * np.pi * 20 * t + phase_offset * 2.5)
            gamma = 0.15 * np.sin(2 * np.pi * 40 * t + phase_offset * 3)
            
            noise = 0.1 * np.random.randn(n_samples)
            
            emotion_mod = self._get_emotion_modulation()
            
            signal = (delta * emotion_mod['delta'] + 
                     theta * emotion_mod['theta'] + 
                     alpha * emotion_mod['alpha'] + 
                     beta * emotion_mod['beta'] + 
                     gamma * emotion_mod['gamma'] + 
                     noise)
            
            data[:, ch] = signal
            
        return data.astype(np.float32)
    
    def _get_emotion_modulation(self):
        intensity = self.emotion_intensity
        
        base = {'delta': 1.0, 'theta': 1.0, 'alpha': 1.0, 'beta': 1.0, 'gamma': 1.0}
        
        if self.emotion_state == 'neutral':
            return base
        elif self.emotion_state == 'happy':
            return {
                'delta': 1.0,
                'theta': 1.0 - 0.2 * intensity,
                'alpha': 1.0 + 0.5 * intensity,
                'beta': 1.0 + 0.3 * intensity,
                'gamma': 1.0 + 0.4 * intensity
            }
        elif self.emotion_state == 'sad':
            return {
                'delta': 1.0 + 0.3 * intensity,
                'theta': 1.0 + 0.4 * intensity,
                'alpha': 1.0 - 0.3 * intensity,
                'beta': 1.0 - 0.2 * intensity,
                'gamma': 1.0 - 0.1 * intensity
            }
        elif self.emotion_state == 'angry':
            return {
                'delta': 1.0,
                'theta': 1.0,
                'alpha': 1.0 - 0.4 * intensity,
                'beta': 1.0 + 0.6 * intensity,
                'gamma': 1.0 + 0.5 * intensity
            }
        elif self.emotion_state == 'surprised':
            return {
                'delta': 1.0 - 0.2 * intensity,
                'theta': 1.0 + 0.3 * intensity,
                'alpha': 1.0 + 0.4 * intensity,
                'beta': 1.0 + 0.2 * intensity,
                'gamma': 1.0 + 0.6 * intensity
            }
        elif self.emotion_state == 'fearful':
            return {
                'delta': 1.0 + 0.2 * intensity,
                'theta': 1.0 + 0.5 * intensity,
                'alpha': 1.0 - 0.3 * intensity,
                'beta': 1.0 + 0.4 * intensity,
                'gamma': 1.0 + 0.3 * intensity
            }
        elif self.emotion_state == 'disgusted':
            return {
                'delta': 1.0 + 0.1 * intensity,
                'theta': 1.0 + 0.2 * intensity,
                'alpha': 1.0 - 0.2 * intensity,
                'beta': 1.0 + 0.3 * intensity,
                'gamma': 1.0
            }
        
        return base


class LSLEEGSource(EEGSource):
    def __init__(self, stream_name=None):
        self._stream_name = stream_name
        self._inlet = None
        self._n_channels = None
        self._sample_rate = None
        self._buffer = []
        
    @property
    def n_channels(self):
        return self._n_channels
    
    @property
    def sample_rate(self):
        return self._sample_rate
        
    def start(self):
        try:
            from pylsl import StreamInlet, resolve_stream
            
            if self._stream_name:
                streams = resolve_stream('name', self._stream_name)
            else:
                streams = resolve_stream('type', 'EEG')
                
            if not streams:
                raise RuntimeError("No EEG stream found")
                
            self._inlet = StreamInlet(streams[0])
            info = self._inlet.info()
            self._n_channels = info.channel_count()
            self._sample_rate = info.nominal_srate()
            
        except ImportError:
            raise RuntimeError("pylsl not installed. Install with: pip install pylsl")
            
    def stop(self):
        if self._inlet:
            self._inlet.close_stream()
            self._inlet = None
            
    def get_sample(self, n_samples=7):
        if not self._inlet:
            raise RuntimeError("Stream not started")
            
        samples = []
        for _ in range(n_samples):
            sample, _ = self._inlet.pull_sample(timeout=1.0)
            if sample:
                samples.append(sample)
                
        if len(samples) < n_samples:
            while len(samples) < n_samples:
                samples.append(samples[-1] if samples else [0] * self._n_channels)
                
        return np.array(samples, dtype=np.float32)


class FileEEGSource(EEGSource):
    def __init__(self, filepath, n_channels=30, sample_rate=256):
        self.filepath = filepath
        self._n_channels = n_channels
        self._sample_rate = sample_rate
        self.data = None
        self.index = 0
        
    @property
    def n_channels(self):
        return self._n_channels
    
    @property
    def sample_rate(self):
        return self._sample_rate
        
    def start(self):
        ext = self.filepath.lower().split('.')[-1]
        
        if ext == 'npy':
            self.data = np.load(self.filepath)
        elif ext == 'npz':
            loaded = np.load(self.filepath)
            self.data = loaded[list(loaded.keys())[0]]
        elif ext == 'csv':
            self.data = np.loadtxt(self.filepath, delimiter=',')
        elif ext == 'pt':
            import torch
            self.data = torch.load(self.filepath).numpy()
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)
        if self.data.shape[1] > self.data.shape[0]:
            self.data = self.data.T
            
        self._n_channels = self.data.shape[1]
        self.index = 0
        
    def stop(self):
        self.data = None
        self.index = 0
        
    def get_sample(self, n_samples=7):
        if self.data is None:
            raise RuntimeError("File not loaded")
            
        end_idx = self.index + n_samples
        
        if end_idx > len(self.data):
            self.index = 0
            end_idx = n_samples
            
        samples = self.data[self.index:end_idx]
        self.index = end_idx
        
        return samples.astype(np.float32)


class BrainFlowEEGSource(EEGSource):
    BOARD_IDS = {
        'synthetic': -1,
        'muse_2': 22,
        'muse_s': 21,
        'openbci_cyton': 0,
        'openbci_ganglion': 1,
        'openbci_cyton_daisy': 2,
        'neurosity_crown': 26,
        'emotiv_epoc': 10,
    }
    
    def __init__(self, board_name='synthetic', serial_port=None):
        self.board_name = board_name
        self.serial_port = serial_port
        self.board = None
        self._n_channels = None
        self._sample_rate = None
        
    @property
    def n_channels(self):
        return self._n_channels
    
    @property
    def sample_rate(self):
        return self._sample_rate
        
    def start(self):
        try:
            from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
            from brainflow.data_filter import DataFilter
            
            board_id = self.BOARD_IDS.get(self.board_name, -1)
            
            params = BrainFlowInputParams()
            if self.serial_port:
                params.serial_port = self.serial_port
                
            self.board = BoardShim(board_id, params)
            self.board.prepare_session()
            self.board.start_stream()
            
            self._sample_rate = BoardShim.get_sampling_rate(board_id)
            eeg_channels = BoardShim.get_eeg_channels(board_id)
            self._n_channels = len(eeg_channels)
            self._eeg_channels = eeg_channels
            
        except ImportError:
            raise RuntimeError("brainflow not installed. Install with: pip install brainflow")
            
    def stop(self):
        if self.board:
            self.board.stop_stream()
            self.board.release_session()
            self.board = None
            
    def get_sample(self, n_samples=7):
        if not self.board:
            raise RuntimeError("Board not started")
            
        data = self.board.get_current_board_data(n_samples)
        eeg_data = data[self._eeg_channels, :].T
        
        if len(eeg_data) < n_samples:
            padding = np.zeros((n_samples - len(eeg_data), self._n_channels))
            eeg_data = np.vstack([eeg_data, padding])
            
        return eeg_data.astype(np.float32)


def create_eeg_source(source_type, **kwargs):
    if source_type == 'simulated':
        return SimulatedEEGSource(**kwargs)
    elif source_type == 'lsl':
        return LSLEEGSource(**kwargs)
    elif source_type == 'file':
        return FileEEGSource(**kwargs)
    elif source_type == 'brainflow':
        return BrainFlowEEGSource(**kwargs)
    else:
        raise ValueError(f"Unknown source type: {source_type}")
