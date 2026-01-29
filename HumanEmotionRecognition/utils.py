import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import torch

def generate_commercial_report(results, output_dir, mode="all_classes"):
    """
    Converts raw model outputs into Business-Ready CSVs and Visual Plots.
    """
    print("\n[INFO] Generating Commercial Reports...")

    # Unpack results (Assuming results structure based on your code)
    # results usually contains: (true_labels, predicted_labels) or similar
    # We will aggregate them into one big dataframe

    all_true = []
    all_pred = []

    # Handle the specific format of your results list
    # Assuming result items are dictionaries or tuples from zeroshot_training return
    for res in results:
        # Adjust indices based on exactly what zeroshot_training returns
        # If it returns (pred_labels, true_labels, etc...)
        if isinstance(res, tuple) or isinstance(res, list):
            # Example: assuming last two items are preds and true
            # You might need to adjust this index based on Zeroshot_setting.py return
            preds = res[0]  # Adjust index if needed
            true = res[1]  # Adjust index if needed

            if isinstance(preds, torch.Tensor): preds = preds.cpu().numpy()
            if isinstance(true, torch.Tensor): true = true.cpu().numpy()

            all_pred.extend(preds)
            all_true.extend(true)

    # 1. Create a CSV Report (Excel Friendly)
    df = pd.DataFrame({
        'True_Emotion': all_true,
        'Predicted_Emotion': all_pred
    })

    # Map numbers to names for readability
    emotion_map = {0: 'Neutral', 1: 'Sadness', 2: 'Anger', 3: 'Happiness', 4: 'Calmness'}
    df['True_Label'] = df['True_Emotion'].map(emotion_map)
    df['Predicted_Label'] = df['Predicted_Emotion'].map(emotion_map)

    csv_path = os.path.join(output_dir, f"{mode}_detailed_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"   -> Saved CSV Report: {csv_path}")

    # 2. Generate Confusion Matrix Plot (Visual)
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_true, all_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_map.values(),
                yticklabels=emotion_map.values())
    plt.title(f'Emotion Recognition Performance ({mode})')
    plt.ylabel('Actual Emotion')
    plt.xlabel('Predicted Emotion')

    img_path = os.path.join(output_dir, f"{mode}_confusion_matrix.png")
    plt.savefig(img_path)
    plt.close()
    print(f"   -> Saved Visualization: {img_path}")

    # 3. Generate Classification Summary (Text)
    report = classification_report(all_true, all_pred, target_names=emotion_map.values())
    txt_path = os.path.join(output_dir, f"{mode}_summary.txt")
    with open(txt_path, "w") as f:
        f.write(report)
    print(f"   -> Saved Summary Text: {txt_path}")