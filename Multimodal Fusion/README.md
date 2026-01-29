# Multimodal Emotion Recognition (EEG + Video)

Research- and production-ready multimodal transformer framework for human emotion recognition.
Supports EEG and video modalities with bottleneck-based fusion.

## Features
- Modular EEG and Video encoders
- Multimodal Bottleneck Transformer (MBT)
- Subject-wise evaluation (DEAP-style)
- Config-driven training

## Installation
```bash
pip install -r requirements.txt
```

## Training
```bash
python src/training/train.py --config configs/deap.yaml
```

## License
MIT
