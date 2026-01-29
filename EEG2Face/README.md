# EEG to FLAME 3D Face Generator

A Windows desktop application that collects EEG signals, processes them using a neural network encoder, and generates 3D facial expressions using a FLAME-based model.

## Features

- **Multiple EEG Sources**: Support for simulated EEG, LSL streams, BrainFlow devices, and file input
- **Real-time Processing**: Live EEG to facial expression mapping
- **3D Visualization**: Interactive PyVista-based 3D face rendering
- **Expression Control**: Adjustable shape parameters and expression scaling
- **Export Options**: Save meshes as OBJ/PLY files, export parameters as NPZ
- **Model Persistence**: Save and load trained model weights

## Installation

### Prerequisites

- Python 3.8 or higher
- Windows 10/11

### Quick Install

1. Double-click `install.bat` to create a virtual environment and install dependencies
2. Run `run.bat` to start the application

### Manual Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Usage

### EEG Collection

1. **Select Source**: Choose from Simulated, LSL Stream, BrainFlow, or File
2. **Configure**: For simulated mode, select emotion and intensity
3. **Start Collection**: Click "Start Collection" to begin real-time processing

### Face Parameters

- **Shape**: Adjust base face shape
- **Expression Scale**: Control expression intensity
- **Auto-update**: Toggle automatic visualization updates
- **Smooth transitions**: Enable smooth morphing between expressions

### Rendering Options

- **Face Color**: Choose between Peach, Gray, Blue, or Custom
- **Wireframe**: Show mesh edges
- **Smooth Shading**: Enable Phong shading

### Export

- **OBJ**: Standard 3D mesh format
- **PLY**: Polygon file format with vertex data
- **Parameters**: NumPy archive with expression/pose parameters

## Supported EEG Devices (via BrainFlow)

- OpenBCI Cyton/Ganglion
- Muse 2/S
- Neurosity Crown
- Emotiv EPOC
- Any LSL-compatible device

## Architecture

```
EEG Data → EEG Encoder (CNN + Attention) → Expression/Pose Parameters → FLAME Model → 3D Mesh
```

### Components

- `main.py`: Main application window and GUI
- `eeg_sources.py`: EEG data source implementations
- `flame_model.py`: FLAME face model and EEG encoder

## Requirements

- PyQt5
- PyTorch
- PyVista
- NumPy

Optional (for real EEG):
- pylsl (LSL streams)
- brainflow (BrainFlow devices)

## License

Research/Educational Use Only

## Acknowledgments

- FLAME: Faces Learned with an Articulated Model and Expressions
- EMOCA: Emotion Driven Monocular Face Capture and Animation
