# 🧠 CogniFace: Integrated BCI & Emotion AI Suite

This repository contains official software deliverables from the grant project:
**“Development of Brain–Computer Interface (BCI) SW/HW Solutions”** (Commercialization Projects Funding)

It includes:

1. **AMBT** – Adaptive Multimodal Bottleneck Transformer for EEG–Audio–Video emotion recognition.
2. **EEG2Face** – EEG-to-FLAME 3D facial expression generator.
3. **HumanEmotionRecognition** – Commercial-grade multimodal BCI emotion recognition system.
4. **CogniFace** – Integrated BCI & Emotion AI Suite for healthcare & rehabilitation.

---

## 1️⃣ Adaptive Multimodal Bottleneck Transformer (AMBT)

**AMBT** fuses **EEG, audio, and facial video** via **adaptive bottleneck-token interaction**, achieving robust emotion recognition in multimodal BCI settings.

### Overview

Facial and speech expressions provide strong cues for emotion recognition, while EEG supplies complementary neural information when external signals are ambiguous or missing.

**AMBT** introduces **Cross-Modal Adaptation Modules (CMAMs)** that enable controlled cross-modal interaction using bottleneck tokens.

<p align="center">
  <img src="Multimodal Fusion/ambt_overview.png" alt="AMBT Overview" width="820"/>
</p>

> **Figure 1.** AMBT overview: unimodal encoders + CMAM adapters with bottleneck tokens enabling cross-modal exchange.

### Results (Benchmarks)

| Dataset     | Modalities          | Accuracy  |
| ----------- | ------------------- | --------- |
| **EAV**     | EEG + Audio + Video | **85.1%** |
| **CREMA-D** | Audio + Video       | **90.9%** |
| **DEAP**    | EEG + Video         | **98.7%** |

### Features

* Independent Transformer encoders for EEG, audio, and video
* CMAM modules for controlled bottleneck-token fusion
* Parameter-efficient (<1% additional trainable parameters)
* Unified latent space for multimodal fusion
* Supports DEAP, EAV, CREMA-D datasets
* Modular codebase (`src/models/`, `src/datasets/`, `src/training/`, `src/utils/`)

---

## 2️⃣ EEG2Face: FLAME 3D Face Generator

**EEG2Face** is a Windows desktop application that **reconstructs 3D facial expressions from EEG signals** using a **FLAME-based model**, guided by a neural network architecture combining **EEG Regressor (𝐸𝑟)**, **Emotion Encoder (𝐸𝑒)**, and pretrained **Vision Encoder (𝐸𝑣)**.
### Overview
<p align="center">
  <img src="EEG2Face/framework.png" alt="EEG2Face Framework" width="820"/>
</p>

> **Figure 2.** EEG2Face framework overview. Training uses contrastive, transformation, and vertex-alignment losses. Inference supports structure-only, natural emotion, and emotion-controlled synthesis.

<p align="center">
  <img src="EEG2Face/comparison.png" alt="EEG2Face Comparison" width="820"/>
</p>

> **Figure 3.** Comparison of ground-truth video frames (GT) with EEG2Face reconstructed expressions.

### Features

* Multiple EEG sources: simulated, LSL streams, BrainFlow devices, file input
* Real-time EEG-to-face mapping
* Interactive 3D visualization via PyVista
* Expression control: adjustable shape parameters, intensity scaling
* Export options: OBJ/PLY meshes, NPZ parameters
* Model persistence: save/load trained weights


### Architecture

```
EEG Data → EEG Encoder (CNN + Attention) → Expression/Pose Parameters → FLAME Model → 3D Mesh
```

### Components

* `main.py`: GUI & main application
* `eeg_sources.py`: EEG data sources
* `flame_model.py`: FLAME face model + EEG encoder

---

## 3️⃣ HumanEmotionRecognition: Multimodal BCI System

**A commercial-grade Brain-Computer Interface (BCI) solution for emotion recognition using EEG, Audio, and Video data.**

### Overview

This software provides a modular AI framework for recognizing human emotional states (**Neutral, Sadness, Anger, Happiness, Calmness**). It leverages a novel **Cross-Modal Bottleneck Fusion** architecture to integrate physiological signals (EEG) with audiovisual cues, enabling high-accuracy emotion detection even in **Zero-Shot** scenarios.

This technology is designed for integration into:

* Medical Rehabilitation Centers
* Interactive Digital Healthcare Systems
* Neuromarketing & User Experience Analytics

<p align="center">
  <img src="HumanEmotionRecognition/model.png" alt="Overview" width="820"/>
</p>

> **Figure 4.** Multimodal ZSL for EEG emotion recognition: EEG is encoded by an EEG Transformer, while audio–video features are fused via bottleneck AV Transformer. Both embeddings are aligned in a shared semantic space for unseen-class prediction using centroid-based inference.

<p align="center">
  <img src="HumanEmotionRecognition/eeg_emo.png" alt="Overview" width="820"/>
</p>

> **Figure 5.** Beta-band EEG patterns across emotions. Averaged 11–20 Hz envelope power over time (20 s speaking task) is shown for five emotions across left hemisphere, frontal lobe, and right hemisphere; topographic maps below illustrate emotion-specific spatial EEG distributions.

### Key Features

* **Multimodal Fusion:** Synchronizes EEG, Audio, and Video streams using a Transformer-based architecture.
* **Zero-Shot Learning:** Recognizes emotional states without subject-specific calibration.
* **Modular Design:** Independent processing blocks for easy integration into existing hospital systems.

### Architecture

* **Inputs:** EEG signals, Audio spectrograms, Video frames
* **Encoders:** Independent Transformer backbones for each modality
* **Fusion:** A shared bottleneck mechanism enables efficient information exchange across modalities before classification

---

## 4️⃣ BCI Suite

### Next-Generation Neuro-Technology for Healthcare & Rehabilitation

### Overview

BCI Suite is a comprehensive Brain-Computer Interface (BCI) solution developed under the **Health & Medical Technology R&D Program**. It bridges the gap between advanced AI research and practical clinical application.

This repository contains the source code for **Deliverable 1.2.1: Low-Level BCI Software Outputs**, featuring:

1. **Zero-Calibration Engine:** Plug-and-play BCI without subject training.
2. **Neuro-Rehab Trainer:** Gamified neurofeedback for patient therapy.
3. **Real-Time Processing:** Low-latency signal filtering and artifact removal.
4. **Universal Hardware Support:** Drivers for Dry-EEG and clinical headsets.

### Commercial Modules

| Module                         | Function                              | Target User                |
| :----------------------------- | :------------------------------------ | :------------------------- |
| **`modules.zero_calibration`** | AI that works instantly on new users  | Hospitals / Public Clinics |
| **`modules.rehab_training`**   | Gamified focus/relaxation training    | Rehab Centers              |
| **`modules.realtime`**         | < 10ms Signal Processing Pipeline     | Hardware Developers        |
| **`modules.dry_eeg`**          | Driver support for next-gen wearables | Device Manufacturers       |


---

## 📚 Publications & Patents

### Publications

1. **M.H. Lee et al.**, *EAV: EEG–Audio–Video Dataset for Emotion Recognition in Conversations*, Scientific Data, 2025
2. **Adaptive Bottleneck Transformer for Multimodal EEG, Audio, and Vision Fusion**, Expert Systems with Applications
3. **EEG2FACE: EEG-driven Emotional 3D Face Reconstruction**, Information Fusion
4. **EEG-Based Emotion Recognition via Vision-Guided Contrastive Learning and Transformer Fusion**, Biomedical Signal Processing and Control

### Patents

* **10465** — Multimodal EEG–Audio–Visual Data Fusion (Granted)
* **11561** — 3D Facial Avatars from EEG/EMG Signals (Domestic Granted; U.S. Pending)

---

## 📩 Contact

**Minho Lee** (PI) — [minho.lee@nu.edu.kz](mailto:minho.lee@nu.edu.kz)
Nazarbayev University, SEDS
