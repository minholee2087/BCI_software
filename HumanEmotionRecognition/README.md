\# HumanEmotionRecognition: Multimodal BCI System



\*\*A commercial-grade Brain-Computer Interface (BCI) solution for emotion recognition using EEG, Audio, and Video data.\*\*



\## 📌 Overview

This software provides a modularized AI framework for recognizing human emotional states (Neutral, Sadness, Anger, Happiness, Calmness). It leverages a novel \*\*Cross-Modal Bottleneck Fusion\*\* architecture to integrate physiological signals (EEG) with audiovisual cues, enabling high-accuracy emotion detection even in Zero-Shot scenarios.



This technology is designed for integration into:

\- Medical Rehabilitation Centers

\- Interactive Digital Healthcare Systems

\- Neuromarketing \& User Experience Analytics



\## 🚀 Key Features

\- \*\*Multimodal Fusion:\*\* Synchronizes EEG, Audio, and Video streams using Transformer-based architecture.

\- \*\*Zero-Shot Learning:\*\* Capable of recognizing emotional states without subject-specific calibration.

\- \*\*Modular Design:\*\* Independent processing blocks for easy integration into existing hospital systems.



\## 🛠️ Installation



1\. \*\*Clone the repository:\*\*

   ```bash

   git clone \[https://github.com/minholee2087/BCI\_software.git](https://github.com/minholee2087/BCI\_software.git)

   cd HumanEmotionRecognition



2\. Install dependencies:

&nbsp;  ```bash
   pip install -r requirements.txt



3\. Download the Dataset: The system requires the Global Multimodal BCI Dataset. Run the following command to download it automatically:

   ```bash
   python main.py --mode download



4\. Usage: The system operates via a command-line interface (CLI) for robust automation.

&nbsp;	1) Run Full Training (All Classes). Train the model on all 5 emotion classes.

&nbsp;	```bash

&nbsp;	python main.py --mode train\_all

&nbsp;	

&nbsp;	2) Run Zero-Shot Experiment. Test the model's ability to recognize a specific emotion (e.g., Happiness) without prior training on that specific class.

&nbsp;	```bash

&nbsp;	python main.py --mode zeroshot --class\_label 3

&nbsp;	# Class Labels: 0=Neutral, 1=Sadness, 2=Anger, 3=Happiness, 4=Calmness



5\. Architecture:

The system utilizes a Cross-Modal Bottleneck Fusion Transformer.

Inputs: EEG signals, Audio spectrograms, Video frames.

Encoders: Independent ViT (Vision Transformer) backbones for each modality.

Fusion: A shared bottleneck mechanism allows modalities to exchange information efficiently before classification.



6\. License: 
This project is licensed under the MIT License - see the LICENSE file for details.



7\. Acknowledgments:

Supported by the Health and Medical Technology R\&D Program and developed in collaboration with Nazarbayev University and DGMIF.

