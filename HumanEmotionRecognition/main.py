import argparse
import sys
import os
from config import check_data_exists, DATA_DIR
from main_classwise import train_zero_shot, train_all_classes
import gdown

# Google Drive ID for the dataset (Extracted from your original link)
GDRIVE_URL = "https://drive.google.com/drive/folders/1gjaHu0Qy6UdBgF1JZ0X7aET30vRUE7qP?usp=sharing"


def download_data():
    """Downloads dataset if not present."""
    print(f"\n[INFO] Downloading data to {DATA_DIR}...")
    try:
        gdown.download_folder(GDRIVE_URL, output=DATA_DIR, quiet=False, use_cookies=False)
        print("\n[SUCCESS] Data download complete.")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="HumanEmotionRecognition: BCI Multimodal Emotion Recognition System",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['download', 'train_all', 'zeroshot'],
        required=True,
        help="Select operation mode:\n"
             "  download   - Download required datasets\n"
             "  train_all  - Train and test on all 5 emotion classes\n"
             "  zeroshot   - Run Zero-Shot learning experiment"
    )

    parser.add_argument(
        '--class_label',
        type=int,
        choices=[0, 1, 2, 3, 4],
        help="Target emotion class for Zero-Shot mode:\n"
             "  0: Neutral\n  1: Sadness\n  2: Anger\n  3: Happiness\n  4: Calmness",
        default=None
    )

    args = parser.parse_args()

    # --- Mode: Download ---
    if args.mode == 'download':
        download_data()
        return

    # --- Check Data Availability ---
    if not check_data_exists():
        print("\n[ERROR] Data not found!")
        print(f"Expected data at: {DATA_DIR}")
        print("Please run: python main.py --mode download")
        sys.exit(1)

    # --- Mode: Train All ---
    if args.mode == 'train_all':
        print("\n[INFO] Starting Full Training on all classes...")
        # Ensure your main_classwise.py has a function for this, or call the script
        # train_all_classes()
        pass

        # --- Mode: Zero-Shot ---
    elif args.mode == 'zeroshot':
        if args.class_label is None:
            print("\n[ERROR] --class_label is required for zeroshot mode.")
            sys.exit(1)

        print(f"\n[INFO] Starting Zero-Shot experiment for Class {args.class_label}...")
        # Ensure main_classwise.py has a function like this:
        # train_zero_shot(args.class_label)
        pass


if __name__ == "__main__":
    main()