import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split

def is_blank(frame):
    return np.all(frame == 0)

def has_blank_frame(clip):
    return any(is_blank(frame) for frame in clip)

# Load all parts
import glob
file_list = sorted(glob.glob("images_part_*.npz"))

all_images = []
all_labels = []
all_filenames = []
bad_indices = []
clip_idx = 0

for file in file_list:
    data = np.load(file, allow_pickle=True)
    images = data["images"]
    labels = data["labels"]
    filenames = data["filenames"]
    print(filenames[0])
    print(labels[0])
    for i, clip in enumerate(images):
        if has_blank_frame(clip):
            bad_indices.append(clip_idx)
            print(filenames[i])
        else:
            all_images.append(clip)
            all_labels.append(labels[i])
            all_filenames.append(filenames[i])
        clip_idx += 1

# Convert to arrays
X_cleaned = np.array(all_images)
y_cleaned = np.array(all_labels)
names_cleaned = np.array(all_filenames)

print(f"Removed {len(bad_indices)} bad clips")
print(f"Final shape: {X_cleaned.shape}")



print("After cleaning:")
print("X_cleaned shape:", X_cleaned.shape)
print("Length y:", len(y_cleaned))
print("Length names:", len(names_cleaned))

# Step 4: Train-test split
indices = np.arange(len(X_cleaned))
train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y_cleaned, random_state=42)

tr_x = [X_cleaned[i] for i in train_idx]
tr_y = [y_cleaned[i] for i in train_idx]
te_x = [X_cleaned[i] for i in test_idx]
te_y = [y_cleaned[i] for i in test_idx]
tr_names = [names_cleaned[i] for i in train_idx]
te_names = [names_cleaned[i] for i in test_idx]

# Example outputs
print("Example train name:", tr_names[1])
print("Example train label:", tr_y[1])

# Step 5: Save final dataset
os.makedirs("D:/Crema_d/Video", exist_ok=True)
file_out = os.path.join("D:/Crema_d/Video", "subject_all_vid_loso.pkl")
with open(file_out, "wb") as f:
    pickle.dump([tr_x, tr_y, te_x, te_y, tr_names, te_names], f)

print("✅ Data saved to subject_all_vid.pkl")
 
target_suffix = "1015_DFA_ANG_XX.mp4"

for i, name in enumerate(names_cleaned):
    if name.lower().endswith(target_suffix.lower()):
        print(f"Found at index {i}: {name}")
        break
else:
    print("Clip not found.")
    
# y_cleaned = [y[i] for i in range(len(y)) if i not in bad_indices]
# names_cleaned = [names[i] for i in range(len(names)) if i not in bad_indices]


# print("After cleaning:")
# print("X_cleaned shape:", X_cleaned.shape)
# print("Length y:", len(y_cleaned))
# print("Length names:", len(names_cleaned))

# # Step 4: Train-test split
# indices = np.arange(len(X_cleaned))
# train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y_cleaned, random_state=42)

# tr_x = [X_cleaned[i] for i in train_idx]
# tr_y = [y_cleaned[i] for i in train_idx]
# te_x = [X_cleaned[i] for i in test_idx]
# te_y = [y_cleaned[i] for i in test_idx]
# tr_names = [names_cleaned[i] for i in train_idx]
# te_names = [names_cleaned[i] for i in test_idx]

# # Example outputs
# print("Example train name:", tr_names[1])
# print("Example train label:", tr_y[1])

# # Step 5: Save final dataset
# os.makedirs("D:/Crema_d/Video", exist_ok=True)
# file_out = os.path.join("D:/Crema_d/Video", "subject_all_vid.pkl")
# with open(file_out, "wb") as f:
#     pickle.dump([tr_x, tr_y, te_x, te_y, tr_names, te_names], f)

# print("✅ Data saved to subject_all_vid.pkl")