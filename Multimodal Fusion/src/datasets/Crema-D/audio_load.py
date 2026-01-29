import os
import torchaudio
from torchaudio.transforms import Resample
from transformers import ASTFeatureExtractor
import numpy as np
import IPython.display as ipd
from EAV_datasplit import EAVDataSplit
import Transformer_Audio
import torch
import time
import matplotlib.pyplot as plt


class DataLoadAudio:
    def __init__(self, subject='all', parent_directory=r'D:\crema-d-mirror', target_sampling_rate=16000):
        self.parent_directory = parent_directory
        self.target_sampling_rate = target_sampling_rate
        self.subject = subject
        self.file_path = []
        self.file_emotion = []
        self.file_name = []
        self.feature = None
        self.label = None
        self.label_indexes = None

    def data_files(self):
        subject = 'AudioWAV'
        label_map = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
        path = os.path.join(self.parent_directory, subject)
        
        self.file_emotion = []
        self.file_path = []
        self.file_name = []

        for i in sorted(os.listdir(path)):
            if not i.endswith(".wav"):
                continue
            parts = i.split("_")
            if len(parts) < 3:
                continue
            emotion_code = parts[2]
            label = label_map.get(emotion_code)
            if label is None:
                continue
            self.file_emotion.append(label)
            self.file_path.append(os.path.join(path, i))
            self.file_name.append(i)

    def feature_extraction(self):
        x, y = [], []
        for idx, path in enumerate(self.file_path):
            waveform, sampling_rate = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            if sampling_rate != self.target_sampling_rate:
                resampler = Resample(orig_freq=sampling_rate, new_freq=self.target_sampling_rate)
                resampled_waveform = resampler(waveform).squeeze().numpy()
            else:
                resampled_waveform = waveform.squeeze().numpy()
            x.append(resampled_waveform)
            y.append(self.file_emotion[idx])
            
        self.feature = x
        self.label_indexes = y
        self.label = y

    def process(self):
        self.data_files()
        self.feature_extraction()
        return self.feature, self.label_indexes, self.file_name


import os

def check_trial_overlap(tr_names, te_names):
    """
    Check if training and testing sets share any trials.
    
    Args:
        tr_names (list of str): List of training file paths.
        te_names (list of str): List of testing file paths.
        
    Returns:
        overlap_count (int): Number of overlapping trials.
        overlap_trials (list of str): Names of overlapping trials.
    """
    # Extract just filenames without directories
    tr_trials = set(os.path.basename(f) for f in tr_names)
    te_trials = set(os.path.basename(f) for f in te_names)
    
    # Find intersection
    overlap_trials = list(tr_trials & te_trials)
    overlap_count = len(overlap_trials)
    
    if overlap_count > 0:
        print(f"Warning: {overlap_count} overlapping trials found!")
        print("Overlapping trials:", overlap_trials)
    else:
        print("No overlapping trials found between training and testing sets.")
    
    return overlap_count, overlap_trials

import pickle
if __name__ == "__main__":
    aud_loader = DataLoadAudio(parent_directory=r'D:\crema-d-mirror')
    
    with open("D:/Crema_d/Audio/subject_all_aud.pkl", "rb") as f:
        tr_x_vis, tr_y_vis, te_x_vis, te_y_vis = pickle.load(f)
        
    print(len(tr_y_vis))
    print(len(te_y_vis))
    print(tr_y_vis[:10])
    
    with open("D:/Crema_d/Video/subject_all_vid.pkl", "rb") as f:
        tr_x_vis, tr_y_vis, te_x_vis, te_y_vis, tr_names, te_names = pickle.load(f)
    print(len(tr_y_vis))
    print(len(te_y_vis))
    print(tr_y_vis[:10])
    print(tr_names[:10])
    check_trial_overlap(te_names, te_names)
    
    # X, y, names = aud_loader.process()
    # file_ = os.path.join("D:/Crema_d/Audio", "subject_all_audio_presplit.pkl")
    # with open(file_, "wb") as f:
    #     pickle.dump([X, y, names], f)
    # indices = np.arange(len(X))
    # train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
    
    # tr_x_aud = [X[i] for i in train_idx]
    # tr_y_aud = [y[i] for i in train_idx]
    # te_x_aud = [X[i] for i in test_idx]
    # te_y_aud = [y[i] for i in test_idx]
    # tr_names = [names[i] for i in train_idx]
    # te_names = [names[i] for i in test_idx]
    
    # os.makedirs("D:/Crema_d/Audio", exist_ok=True)
    # file_ = os.path.join("D:/Crema_d/Audio", "subject_all_aud.pkl")
    # with open(file_, "wb") as f:
    #     pickle.dump([tr_x_aud, tr_y_aud, te_x_aud, te_y_aud, tr_names, te_names], f)
    
    
    # data = [tr_x_aud, tr_y_aud, te_x_aud, te_y_aud]
    # # print(data[0].size())
    # # print(data[2].shape)
    
    # # print(tr_x_aud.shape)
    # # print(te_x_aud.shape)
    # # for i, sample in enumerate(data[2]):
    # #     print(f"Sample {i} shape: {sample.shape}")
    # mod_path = r"D:\Downloads\ast-finetuned-audioset-10-10-0.4593"
    # Trainer = Transformer_Audio.AudioModelTrainer(data,  model_path=mod_path, num_classes=6, weight_decay=1e-5, lr=0.005, batch_size = 8)

    # Trainer.train(epochs=10, lr=5e-4, freeze=True)
    # Trainer.train(epochs=15, lr=5e-6, freeze=False)

    # test_acc.append(Trainer.outputs_test)

   



''' test it with the current data

model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

test_data = torch.tensor(data, dtype=torch.float32)
test_data = test_data.to(device)
aa = test_data[0:20]
with torch.no_grad(): # 572 classes. 
    logits = model(aa).logits

probs = torch.nn.functional.softmax(logits, dim=-1)
predicted_class_id = probs.argmax(dim=1)
bb = np.array(probs.cpu())
config = model.config
config.num_labels
'''
