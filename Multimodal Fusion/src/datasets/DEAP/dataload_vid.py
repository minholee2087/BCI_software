import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch
from sklearn.model_selection import train_test_split
import pickle
import re
import pandas as pd



def extract_trial_number(filename):
    match = re.search(r'trial(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

class DataLoadVision:
    def __init__(self, subject='all', parent_directory=r'D:\Downloads\face_video', face_detection=True,
                 image_size=224, label_file_path=r"D:\Downloads\metadata_xls\participant_ratings.xls"):
        self.IMG_HEIGHT, self.IMG_WIDTH = 360,480
        self.subject = subject
        self.parent_directory = parent_directory
        self.file_path = list()
        self.images  = list()
        self.labels_val_y = list()
        self.labels_aro_y = list()
        self.subjects = list()
        self.names = list()
        self.face_detection = face_detection
        self.image_size = image_size
        self.face_image_size = 56  #
        self.label_df = pd.read_excel(label_file_path)
        self.label_df.columns = self.label_df.columns.str.strip()  # Clean headers
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(
            image_size=self.face_image_size, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )

    def data_files(self):
        subject_id = int(self.subject)  # Convert subject string to int
        subject = f"s{subject_id:02d}"
        path = os.path.join(self.parent_directory, subject)
        sorted_files = sorted(os.listdir(path), key=extract_trial_number)
    
        for filename in sorted_files:
            trial_match = re.search(r'trial(\d+)', filename)
            if trial_match:
                trial_number = int(trial_match.group(1))
    
                # Get valence and arousal from DataFrame
                row = self.label_df[
                    (self.label_df["Participant_id"] == subject_id) &
                    (self.label_df["Trial"] == trial_number)
                ]
    
                if not row.empty:
                    valence = row["Valence"].values[0]
                    arousal = row["Arousal"].values[0]
    
                    # Binarize valence and arousal (0/1)
                    self.labels_val_y.append(1 if valence > 5 else 0)
                    self.labels_aro_y.append(1 if arousal > 5 else 0)
                else:
                    # Handle missing row (you may also choose to skip this trial)
                    self.labels_val_y.append(None)
                    self.labels_aro_y.append(None)
            
                self.names.append(filename)
                self.file_path.append(os.path.join(path, filename))

        

    # def data_load(self, train_idx, test_idx):
    #     video_counter = 0
    #     save_counter = 0
    #     self.labels = []
    #     self.filenames = []
        
    #     train_files = [self.file_path[i] for i in train_idx]

    #     test_files = [self.file_path[i] for i in test_idx]

        
        
        
    #     for idx, file in zip(train_idx, train_files):
    #         # if idx>1:
    #         #     break
    #         print(file)
    #         cap = cv2.VideoCapture(file)
    #         # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # ~600
    #         # frame_rate = cap.get(cv2.CAP_PROP_FPS) # 30 frame
    #         a1 = []
    #         if cap.isOpened():
    #             frame_index = 1
    #             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
    #             while True:
    #                 ret, frame = cap.read()
    #                 if not ret:
    #                     break
    #                 if (frame_index - 1) % 5 == 0 and frame_index <= 3000:
    #                     if self.face_detection:
    #                         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                         with torch.no_grad():
    #                             x_aligned, prob = self.mtcnn(frame, return_prob=True)
    #                             if prob is not None and prob > 0.3:
    #                                 x_aligned = (x_aligned + 1) / 2
    #                                 x_aligned = np.clip(x_aligned * 255, 0, 255)
    #                                 x_aligned = np.transpose(x_aligned.numpy().astype('uint8'), (1, 2, 0))
    #                                 a1.append(x_aligned)
    #                             else:
    #                                 print(f"Face is not detected in file {file}, original is saved")
    #                                 blank_img = np.zeros((56, 56, 3), dtype=np.uint8)
    #                                 a1.append(blank_img)

    #                             pass
    #                     else:
    #                         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                         resizedImg = cv2.resize(frame, (self.image_size, self.image_size))
    #                         a1.append(resizedImg)  # sabina: dlkfjefoie
    
    #                     if len(a1) == 40:  # 20 frame is 2s each
    #                         self.images.append(a1)  # this will contain 400 samples [400, 25, (225, 225, 3)]
    #                         self.labels.append((self.labels_val_y[idx], self.labels_aro_y[idx]))        # e.g., labels[idx]
    #                         self.filenames.append(file)
    #                         a1 = []
    #                 frame_index += 1

    #             cap.release()
            
    #     self.train_images=self.images
    #     self.train_labels=self.labels
    #     self.images = []
    #     self.labels = []

    #     for idx, file in zip(test_idx, test_files):
    #         # if idx>1:
    #         #     break
    #         print(file)
    #         cap = cv2.VideoCapture(file)
    #         # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # ~600
    #         # frame_rate = cap.get(cv2.CAP_PROP_FPS) # 30 frame
    #         a1 = []
    #         if cap.isOpened():
    #             frame_index = 1
    #             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
    #             while True:
    #                 ret, frame = cap.read()
    #                 if not ret:
    #                     break
    #                 if (frame_index - 1) % 5 == 0 and frame_index <= 3000:
    #                     if self.face_detection:
    #                         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                         with torch.no_grad():
    #                             x_aligned, prob = self.mtcnn(frame, return_prob=True)
    #                             if prob is not None and prob > 0.3:
    #                                 x_aligned = (x_aligned + 1) / 2
    #                                 x_aligned = np.clip(x_aligned * 255, 0, 255)
    #                                 x_aligned = np.transpose(x_aligned.numpy().astype('uint8'), (1, 2, 0))
    #                                 a1.append(x_aligned)
    #                             else:
    #                                 print(f"Face is not detected in file {file}, original is saved")
    #                                 blank_img = np.zeros((56, 56, 3), dtype=np.uint8)
    #                                 a1.append(blank_img)

    #                             pass
    #                     else:
    #                         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                         resizedImg = cv2.resize(frame, (self.image_size, self.image_size))
    #                         a1.append(resizedImg)  # sabina: dlkfjefoie
    
    #                     if len(a1) == 40:  # 20 frame is 2s each
    #                         self.images.append(a1)  # this will contain 400 samples [400, 25, (225, 225, 3)]
    #                         self.labels.append((self.labels_val_y[idx], self.labels_aro_y[idx]))        # e.g., labels[idx]
    #                         self.filenames.append(file)
    #                         a1 = []
    #                 frame_index += 1

    #             cap.release()
            
            # video_counter += 1
            # if video_counter == 10 or idx == len(self.file_path) - 1:
            #     save_path = f"images_part_{save_counter}.npz"
            #     np.savez(save_path,
            #              images=np.array(self.images),
            #              labels=np.array(self.labels),
            #              filenames=np.array(self.filenames, dtype=object))
            #     print(f"Saved {len(self.images)} clips to {save_path}")
                
            #     self.images = []
            #     self.labels = []
            #     self.filenames = []
            #     video_counter = 0
            #     save_counter += 1
            
    def data_load(self):
        self.images = []
        self.labels = []
        self.filenames = []
        self.sample_file_idx = []  # keeps track of which file each sample came from
    
        for idx, file in enumerate(self.file_path):
            print(file)
            cap = cv2.VideoCapture(file)
            a1 = []
    
            if cap.isOpened():
                frame_index = 1
    
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
    
                    if (frame_index - 1) % 5 == 0 and frame_index <= 3000:
                        if self.face_detection:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            with torch.no_grad():
                                x_aligned, prob = self.mtcnn(frame, return_prob=True)
                                if prob is not None and prob > 0.3:
                                    x_aligned = (x_aligned + 1) / 2
                                    x_aligned = np.clip(x_aligned * 255, 0, 255)
                                    x_aligned = np.transpose(
                                        x_aligned.numpy().astype("uint8"), (1, 2, 0)
                                    )
                                    a1.append(x_aligned)
                                else:
                                    blank_img = np.zeros((56, 56, 3), dtype=np.uint8)
                                    a1.append(blank_img)
                        else:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            resizedImg = cv2.resize(
                                frame, (self.image_size, self.image_size)
                            )
                            a1.append(resizedImg)
    
                        if len(a1) == 40:
                            self.images.append(a1)
                            self.labels.append(
                                (self.labels_val_y[idx], self.labels_aro_y[idx])
                            )
                            self.filenames.append(file)
                            self.sample_file_idx.append(idx)
                            a1 = []
    
                    frame_index += 1
    
                cap.release()
    def data_split(self, train_idx, test_idx):
        # -------- Split AFTER loading --------
        self.train_images = []
        self.train_labels = []
        self.train_filenames = []
    
        self.test_images = []
        self.test_labels = []
        self.test_filenames = []
    
        train_idx = set(train_idx)
        test_idx = set(test_idx)
    
        for img, lbl, fname, fidx in zip(
            self.images, self.labels, self.filenames, self.sample_file_idx
        ):
            if fidx in train_idx:
                self.train_images.append(img)
                self.train_labels.append(lbl)
                self.train_filenames.append(fname)
            elif fidx in test_idx:
                self.test_images.append(img)
                self.test_labels.append(lbl)
                self.test_filenames.append(fname)
        return self.train_images, self.train_labels, self.test_images, self.test_labels

    def process(self):
        self.data_files()
        self.data_load()
        return self.images, self.labels, self.filenames, self.sample_file_idx

def remove_subject_from_split(x, y, idx_list, REMOVE_SUBJECT):
    """
    x, y: arrays to remove from
    idx_list: numpy array or list of subject indices
    """
    print(idx_list)
    # Convert to numpy
    idx_arr = np.array(idx_list)
    
    # Find where REMOVE_SUBJECT appears
    positions = np.where(idx_arr == REMOVE_SUBJECT)[0]

    if len(positions) > 0:
        i = positions[0]  # subject position
        start = i * BLOCK
        end = (i + 1) * BLOCK

        print(f"Removing subject {REMOVE_SUBJECT} at position {i}, samples {start}:{end}")

        # Remove corresponding EEG samples
        x = np.delete(x, np.s_[start:end], axis=0)
        y = np.delete(y, np.s_[start:end], axis=0)

        # Remove the subject from index list
        idx_arr = np.delete(idx_arr, i)

    return x, y, idx_arr

import torch

# def check_sync(name, y1, y2):
#     """
#     name: 'train' or 'test'
#     y1, y2: lists, tuples, or torch tensors
#     """

#     # Convert torch tensors to list
#     y1_list = [list(item) if isinstance(item, (tuple, list)) else item for item in y1]
#     y2_list = [list(item.numpy()) if torch.is_tensor(item) else (list(item) if isinstance(item, tuple) else item)
#                 for item in y2]

#     # --- Length mismatch ---
#     if len(y1_list) != len(y2_list):
#         raise AssertionError(
#             f"{name} labels length mismatch: {len(y1_list)} (video) vs {len(y2_list)} (EEG)"
#         )

#     # --- Value mismatch ---
#     for i, (a, b) in enumerate(zip(y1_list, y2_list)):
#         if a != b:
#             raise AssertionError(
#                 f"{name} label mismatch at index {i}: "
#                 f"video={a}, eeg={b}"
#             )

#     return True

def check_sync(name, y1, y2):

    # length mismatch
    if len(y1) != len(y2):
        print(f"\n{name} LENGTH MISMATCH: {len(y1)} != {len(y2)}")
        return

    for i, (a, b) in enumerate(zip(y1, y2)):
        a_arr = np.array(a)
        b_arr = np.array(b)
        
        # value mismatch
        if not np.array_equal(a_arr, b_arr):
            print(f"\n{name} VALUE MISMATCH at index {i}:")
            print("y1:", a_arr)
            print("y2:", b_arr)
            return

    print(f"{name}: All good ✓")



if __name__ == '__main__':

    subs=[3,5,11,14]
    for sub in range(1,23):
        vis_loader = DataLoadVision(parent_directory=r'D:\Downloads\face_video',
                                    face_detection=True, subject=sub, label_file_path=r"D:\Downloads\metadata_xls\participant_ratings.xls")
        images, labels, filenames, sample_file_idx=vis_loader.process()
        os.makedirs(r"D:\Deap\Video", exist_ok=True)
        file_ = os.path.join(
            r"D:\Deap\Video", 
            "subject_presplit.pkl"
        )
        with open(file_, "wb") as f:
            pickle.dump([images, labels, filenames, sample_file_idx], f)
            
            
        for fold in range(0,10):
            save_dir = r"D:\Deap\EEG"
            split_path = os.path.join(save_dir, f"subject{sub}_fold{fold}_10fold.pkl")
            with open(split_path, "rb") as f:
                tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg, train_idx, test_idx = pickle.load(f)
                
            print(len(tr_y_eeg))
            print(tr_x_eeg.shape)
            tr_x_eeg = np.array(tr_x_eeg)
            tr_y_eeg = np.array(tr_y_eeg)
            te_x_eeg = np.array(te_x_eeg)
            te_y_eeg = np.array(te_y_eeg)
            if sub==3 or sub==5 or sub==14 or sub==11:
                
                # Convert to numpy for easier slicing
                
                
                BLOCK = 15   # samples per subject
                
                if sub==3 or sub==5 or sub==14 :
                    REMOVE_SUBJECTS = [39]
                else:
                    REMOVE_SUBJECTS = [37,38,39]
                    
                for remove_subject in REMOVE_SUBJECTS:    
                    
                    # Remove from training
                    tr_x_eeg, tr_y_eeg, train_idx = remove_subject_from_split(
                        tr_x_eeg, tr_y_eeg, train_idx, remove_subject
                    )
                    
                    # Remove from testing
                    te_x_eeg, te_y_eeg, test_idx = remove_subject_from_split(
                        te_x_eeg, te_y_eeg, test_idx, remove_subject
                    )
                split_path = os.path.join(save_dir, f"subject{sub}_fold{fold}_10fold_updated.pkl")    
                # Save updated splits if needed
                with open(split_path, "wb") as f:
                    pickle.dump([tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg, train_idx, test_idx], f)
                    
            print(len(tr_y_eeg))
            print(tr_x_eeg.shape)
        
        
            tr_x,tr_y, te_x, te_y = vis_loader.data_split(train_idx, test_idx)
            
            
            # If nothing raised, save the file
            file_ = os.path.join(
                r"D:\Deap\Video", 
                f"subject_sub{sub}_vid_fold{fold}_10fold.pkl"
            )
            with open(file_, "wb") as f:
                pickle.dump([tr_x, tr_y, te_x, te_y], f)
                
            # file_ = os.path.join(
            #     r"D:\Deap\Video", 
            #     f"subject_all_vid_sub{sub}_trialwise8020.pkl"
            # )
            # with open(file_, "rb") as f:
            #     tr_x, tr_y, te_x, te_y = pickle.load(f)
            # print( te_y)
            # print( te_y_eeg)
            check_sync("train", tr_y, tr_y_eeg)
            check_sync("test", te_y, te_y_eeg)
            
            
            with open(file_, "wb") as f:
                pickle.dump([tr_x, tr_y, te_x, te_y], f)
        
                
        # # file_ = os.path.join(r"D:\Deap\Video", f"subject_all_vid_sub{sub}_trialwise_valence.pkl")
        # # with open(file_, "wb") as f:
        # #     pickle.dump([x,y], f)
        
        
        
        # # # print(len(x))
        # # # print(y)
        
        # # # with open(file_, "rb") as f:
        # # #     x,y = pickle.load(f)
        
        # # X=np.stack(x) 
        # # print(X.shape)
        
        # # file_ = os.path.join(r"D:\Deap\EEG", f"subject_all_eeg_sub{sub}_trialwise_valence.pkl")
        # # with open(file_, "rb") as f:
        # #     _, tr_y_eeg, _, te_y_eeg, train_idx, test_idx = pickle.load(f)

        
        # # tr_x = [X[i] for i in train_idx if i < len(X)]
        # # tr_y = [y[i][0] for i in train_idx if i < len(X)]
        # # te_x = [X[i] for i in test_idx if i < len(X)]
        # # te_y = [y[i][0] for i in test_idx if i < len(X)]
        
        
        # # # print(te_y)
        # # # print(te_y_eeg)
        # # # print(train_idx[0])
        # # # print(len(tr_x))
        
        # # os.makedirs(r"D:\Deap\Video", exist_ok=True)
        # # file_ = os.path.join(r"D:\Deap\Video", f"subject_all_vid_sub{sub}_trialwise_arousal.pkl")
        # # with open(file_, "wb") as f:
        # #     pickle.dump([tr_x, tr_y, te_x, te_y], f)
            
            
            
        # # file_ = os.path.join(r"D:\Deap\EEG", f"subject_all_eeg_sub{sub}_trialwise_arousal.pkl")
        # # with open(file_, "rb") as f:
        # #     _, tr_y_eeg, _, te_y_eeg, train_idx, test_idx = pickle.load(f)

        
        # # tr_x = [X[i] for i in train_idx if i < len(X)]
        # # tr_y = [y[i][1] for i in train_idx if i < len(X)]
        # # te_x = [X[i] for i in test_idx if i < len(X)]
        # # te_y = [y[i][1] for i in test_idx if i < len(X)]
        
        
        # # # print(te_y)
        # # # print(te_y_eeg)
        # # # print(train_idx[0])
        # # # print(len(tr_x))
        
        # # os.makedirs(r"D:\Deap\Video", exist_ok=True)
        # # file_ = os.path.join(r"D:\Deap\Video", f"subject_all_vid_sub{sub}_trialwise_arousal.pkl")
        # # with open(file_, "wb") as f:
        # #     pickle.dump([tr_x, tr_y, te_x, te_y], f)
            
            
