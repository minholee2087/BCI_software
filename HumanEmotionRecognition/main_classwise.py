from src.processing.Zeroshot_setting import *
from src.models.Transformer_EEG import *  # Adjusted import to match flat structure
import numpy as np
import torch
import gc
import random
import os
import pickle
from config import INPUT_DATA_DIR, FINETUNED_MODELS_DIR, RESULTS_DIR  # NEW: Import paths from config
from utils import generate_commercial_report

# Weights randomness for consistent results
def set_random_seed(seed):
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # PyTorch multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables cuDNN auto-tuning

# Set the random seed
seed = 42
set_random_seed(seed)

# Check if GPU memory is available
def check_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
        print(
            f"GPU Memory Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024 ** 3:.2f} GB")

# Clear the GPU memory
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# Add memory management function
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# --- UPDATE: Paths are now dynamic using config.py ---
input_pkl_dir = INPUT_DATA_DIR
finetuned_models_dir = FINETUNED_MODELS_DIR
results_path = RESULTS_DIR
# -----------------------------------------------------
results = list()

def train_all_classes():
    # Please choose the number of subjects to use (from 1 to 43) - overall 42 subjects
    # Currently set to range(1, 2) for demo purposes
    for sub in range(1, 2):
        try:
            # Clear memory at the start of each iteration
            clear_memory()

            print(f"Processing subject {sub} ...")
            Data = load_subject_data(directory=input_pkl_dir, subject_idx=sub)
            Models = load_models(base_dir=finetuned_models_dir, subject_idx=sub)

            [tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence,
             te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence] = prepare_multilabel_data(Data[-4:])

            model_aud, model_vis, model_av = Models
            model_zs = ZeroShotModel(eeg_dim=2600, shared_dim=256, num_classes=5)

            # If necessary edit num_layers
            model_eeg = MultiTaskShallowConvNet(Chans=30, Samples=500, num_layers=2)

            # Increase or decrease num_epochs if necessary
            trainer = Trainer_eeg_multitask(
                model=model_eeg,
                data=[tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence,
                      te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence],
                num_epochs=500
            )
            trainer.train()

            out = zeroshot_training(
                Data=Data,
                Models=[model_aud, model_vis, model_eeg, model_av],
                ZeroShotModel=model_zs,
                epochs=100
            )
            results.append(out)

            # Clear memory after each subject
            del model_zs, model_eeg, trainer, Data, Models
            clear_memory()

        except RuntimeError as e:
            print(f"Error processing subject {sub}: {e}")
            clear_memory()
            continue
        except Exception as e:
            print(f"Unexpected error processing subject {sub}: {e}")
            clear_memory()
            continue

    # 1. Save the raw pickle (Backup for devs)
    save_file = os.path.join(results_path, "allclasses_results.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(results, f)

    # 2. Generate the Commercial Assets (For the Grant)
    generate_commercial_report(results, results_path, mode="All_Classes")

    print(f"✅ All results saved to '{results_path}'")

results_zs = list()

def train_zero_shot(class_label):
    # Please choose the number of subjects to use (from 1 to 43)
    # Currently set to range(1, 2) for demo purposes
    for sub in range(1, 2):
        try:
            clear_memory()

            print(f"Processing subject {sub} ...")
            Data = load_subject_data(directory=input_pkl_dir, subject_idx=sub)

            # Exclude chosen class for Zero-Shot setting
            Data_zs = prepare_zeroshot_data(Data, exclude_class=class_label)

            [tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence,
             te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence] = prepare_multilabel_data(Data_zs[-4:])

            Models = load_models(base_dir=finetuned_models_dir, subject_idx=sub)
            model_aud, model_vis, model_av = Models
            model_zs = ZeroShotModel(eeg_dim=2600, shared_dim=256, num_classes=5)

            # If necessary edit num_layers
            model_eeg = MultiTaskShallowConvNet(Chans=30, Samples=500, num_layers=2)

            # Increase or decrease num_epochs if necessary
            trainer = Trainer_eeg_multitask(
                model=model_eeg,
                data=[tr_x_eeg, tr_y_eeg, tr_y_arousal, tr_y_valence,
                      te_x_eeg, te_y_eeg, te_y_arousal, te_y_valence],
                num_epochs=500
            )
            trainer.train()

            out = zeroshot_training(
                Data=Data,
                Models=[model_aud, model_vis, model_eeg, model_av],
                ZeroShotModel=model_zs,
                epochs=100
            )
            results_zs.append(out)

            # Clear memory after each subject
            del model_zs, model_eeg, trainer, Data, Data_zs, Models
            clear_memory()
            clear_gpu_memory()

        except RuntimeError as e:
            print(f"Error processing subject {sub}: {e}")
            clear_memory()
            clear_gpu_memory()
            continue

        except Exception as e:
            print(f"Unexpected error processing subject {sub}: {e}")
            clear_memory()
            clear_gpu_memory()
            continue

    # 1. Save the raw pickle
    save_file = os.path.join(results_path, f"zeroshot_results_class{class_label}.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(results_zs, f)

    # 2. Generate the Commercial Assets
    generate_commercial_report(results_zs, results_path, mode=f"ZeroShot_Class_{class_label}")

    print(f"✅ Results saved to '{results_path}'")