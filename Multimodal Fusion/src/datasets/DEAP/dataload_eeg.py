import numpy as np
from transformers import AutoImageProcessor
from torch.cuda.amp import autocast
from torchvision.models import resnet50
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from timm.layers import Mlp, DropPath, use_fused_attn



import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score



import torch
import torch.nn as nn

from sklearn.utils.class_weight import compute_class_weight

class EEGNet_tor(nn.Module):
    def __init__(self, nb_classes, Chans=30, Samples=500, dropoutRate=0.5, kernLength=300, F1=8, D=8, F2=40,
                  norm_rate=1.0, dropoutType='Dropout'):
        super(EEGNet_tor, self).__init__()

        # Configure dropout
        self.dropout = nn.Dropout(dropoutRate) if dropoutType == 'Dropout' else nn.Dropout2d(dropoutRate)

        # Block 1
        self.firstConv = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.firstBN = nn.BatchNorm2d(F1)
        self.elu = nn.ELU()

        #self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, padding=0, bias=False)
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, padding='valid', bias=False)
        self.depthwiseBN = nn.BatchNorm2d(F1 * D)
        self.depthwisePool = nn.AvgPool2d((1, 50), stride=(1, 7))

        # Applying max-norm constraint
        #self.depthwiseConv.register_forward_hook(
            # lambda module, inputs, outputs: module.weight.data.renorm_(p=2, dim=0, maxnorm=norm_rate))

        # Block 2
        self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.separableBN = nn.BatchNorm2d(F2)
        self.separablePool = nn.AvgPool2d((1, 8))

        # Final layers
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * 67, nb_classes)
        #self.dense2 = nn.Linear(512, 5)
        #print(F2 * ((Samples // 4 // 8)))
        self.softmax = nn.Softmax(dim=1)

        # Applying max-norm constraint
        #self.dense.register_forward_hook(
            #lambda module, inputs, outputs: module.weight.data.renorm_(p=2, dim=0, maxnorm=norm_rate))
    def get_feature_map(self, x):
        x = self.firstConv(x)
        x = self.firstBN(x)
        x = self.elu(x)
        x = self.depthwiseConv(x)
        x = self.depthwiseBN(x)
        x = self.elu(x)
        x = self.depthwisePool(x)
        x = self.dropout(x)
        x = self.separableConv(x)
        x = self.separableBN(x)
        x = self.elu(x)
        x = self.dropout(x)
        return x
    def forward(self, x):
        #print(f"in: {x.shape}")
        x = self.firstConv(x)
        #print(f"in1: {x.shape}")
        x = self.firstBN(x)
        #print(f"in2: {x.shape}")
        x = self.elu(x)
        #print(f"in3: {x.shape}")
        x = self.depthwiseConv(x)
        #print(f"in4: {x.shape}")
        x = self.depthwiseBN(x)
        #print(f"in5: {x.shape}")
        x = self.elu(x)
        #print(f"in6: {x.shape}")
        x = self.depthwisePool(x)
        #print(f"in7: {x.shape}")
        x = self.dropout(x)
        #print(f"in8: {x.shape}")
        x = self.separableConv(x)
        #print(f"in9: {x.shape}")
        x = self.separableBN(x)
        #print(f"in10: {x.shape}")
        x = self.elu(x)
        #print(f"in11: {x.shape}")
        #x = self.separablePool(x)
        #print(f"in12: {x.shape}")
        x = self.dropout(x)
        #print(f"out: {x.shape}")
        import time
        #time.sleep(5)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x

#model = ShallowConvNet(nb_classes=5, Chans=30, Samples=500)
# Generate random data
#data = torch.randn(100, 1, 30, 500)
# Get model output
#output = model(data)
# 40, 1, 227


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_dim):
        super(PatchEmbedding, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_dim = qkv_dim
        self.W_v = nn.ModuleList([nn.Linear(32, 1, bias=False) for _ in range(40)])

    def forward(self, x):
        outputs_V_res = []
        for i in range(40):
            x_head = x[:, i, :, :].permute(0, 2, 1)  # (batch, head_dim, seq_len)
            V = self.W_v[i](x_head)
            outputs_V_res.append(V)
        outputs_V_res = torch.cat(outputs_V_res, dim=-1)
        return outputs_V_res


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_dim):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_dim = qkv_dim

        self.W_q = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.W_k = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.W_v = nn.Linear(self.head_dim, self.qkv_dim, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        outputs = []
        outputs_V_res = []
        for i in range(self.num_heads):
            x_head = x[:, i, :, :]
            Q = self.W_q(x_head)
            K = self.W_k(x_head)
            V = self.W_v(x_head)

            Q = Q.permute(0, 2, 1)
            attn_scores = torch.matmul(Q, K) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)

            attn_output = torch.matmul(V, attn_weights)
            outputs.append(attn_output)
            outputs_V_res.append(V)

        attn_output = torch.cat(outputs, dim=-1)
        outputs_V_res = torch.cat(outputs_V_res, dim=-1)

        return x_head + outputs_V_res  # residual


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim, expansion=4, drop_p=0.5):
        super(FeedForwardBlock, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * expansion)
        self.fc2 = nn.Linear(embed_dim * expansion, embed_dim)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_dim, expansion=4, drop_p=0.5, mid_layer=384):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, qkv_dim)
        self.feed_forward = FeedForwardBlock(embed_dim, expansion, drop_p)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(drop_p)
        
        self.midsample = nn.Linear(40, mid_layer)
        self.endsample = nn.Linear(mid_layer*2, 40)

    def forward(self, x):
        # Attention and residual connection
        attn_output = self.attention(x)
        x = x + self.dropout(self.norm1(attn_output))

        # Feed-forward and residual connection
        ffn_output = self.feed_forward(x)
        x = x + self.dropout(self.norm2(ffn_output))

        return x


class ShallowConvNet(nn.Module):
    def __init__(self, nb_classes, Chans=30, Samples=500, dropoutRate=0.5, num_layers=12,mid_layer=384):
        super(ShallowConvNet, self).__init__()

        self.conv1_depth = 40
        self.eeg_ch = Chans

        # Convolutional block
        self.conv1 = nn.Conv2d(1, self.conv1_depth, (1, 13), bias=False)
        self.pool = nn.AvgPool2d((1, 35), stride=(1, 7))
        self.dropout = nn.Dropout(dropoutRate)

        self.qkv_dim = 40
        self.num_heads = 1
        embed_dim = 40
        self.embedding = PatchEmbedding(embed_dim, self.num_heads, self.qkv_dim)
        self.num_layers=num_layers
        # Transformer Layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim=embed_dim, num_heads=self.num_heads, qkv_dim=self.qkv_dim, drop_p=dropoutRate,mid_layer=mid_layer)
            for _ in range(num_layers)
        ])

        self.batchnorm = nn.BatchNorm2d(40, eps=1e-05, momentum=0.1)
        self.fc = nn.Linear(2680, nb_classes, bias=False)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        batch, channels, height, width = x.shape
        # print(x.shape)
        V = self.embedding(x)
        # print(V.shape)
        for layer in self.transformer_layers:
            V = layer(V)
        # print(V.shape)
        x = V.permute(0, 2, 1).unsqueeze(2)
        # print(x.shape)
        #x = self.batchnorm(x)
        # print(x.shape)
        #x = torch.square(x)
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        #x = torch.log(torch.clamp(x, min=1e-7, max=10000))
        # print(x.shape)
        x = x.squeeze(2)
        # print(x.shape)
        x = self.dropout(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        # import time
        # time.sleep(5)
        x = self.fc(x)
        x = F.softmax(x, dim=1)

        return x
    def forward_ending(self, x):#we take the feature map of the eeg encoded after 4 layers and put it in
        
        x = x.permute(0, 2, 1).unsqueeze(2)
        x = self.batchnorm(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-7, max=10000))

        x = x.squeeze(2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    def feature(self, x):
        print(x.shape)
        x = self.conv1(x)
        batch, channels, height, width = x.shape
        print(x.shape)
        V = self.embedding(x)
        print(V.shape)
        for layer in self.transformer_layers:
            V = layer(V)
        print(V.shape)
        x = V.permute(0, 2, 1).unsqueeze(2)
        print(x.shape)
        x = self.batchnorm(x)
        print(x.shape)
        x = torch.square(x)
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = torch.log(torch.clamp(x, min=1e-7, max=10000))
        print(x.shape)
        x = x.squeeze(2)
        print(x.shape)
        x = self.dropout(x)
        print(x.shape)
        x = torch.flatten(x, 1)
        print(x.shape)
        return x


import os
import pickle
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.model_selection import KFold, StratifiedKFold


# ----------------------------------------------------
# FUNCTION: Segment trials into EEG windows
# ----------------------------------------------------
def segment_trial(eeg_trial, label, window_size, stride):
    """
    eeg_trial: (32, 5120) or more
    label: [valence, arousal]
    returns list of windows and labels
    """
    segments = []
    labels = []

    N = eeg_trial.shape[1]
    for start in range(0, N - window_size + 1, stride):
        window = eeg_trial[:, start:start + window_size]
        segments.append(window)
        labels.append(label)

    return segments, labels


if __name__ == "__main__":   

    # ----------------------------------------------------
    # PARAMETERS
    # ----------------------------------------------------
    window_size = 128 * 4        # 4 seconds
    stride = 128 * 4             # No overlap
    save_dir = r"D:\Deap\EEG"
    os.makedirs(save_dir, exist_ok=True)
    
    xls_path = r'D:\Downloads\metadata_xls\participant_ratings.xls'
    
    


    modes=["val","aro"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # ----------------------------------------------------
    # MAIN LOOP FOR SUBJECT
    # ----------------------------------------------------
    for sub in range(1, 23):   # <-- Change range for all subjects
        # print(f"Processing subject {sub}")
    
        # # Load .mat file
        # data = loadmat(fr'D:\Downloads\data_preprocessed_matlab\data_preprocessed_matlab\s{sub:02d}.mat')
        # X = data['data']      # (40, 40, 8064)
        # labels = data['labels']
    
        # # Load XLS to get correct trial order
        # df = pd.read_excel(xls_path)
        # df = df[df['Participant_id'] == sub].sort_values(by='Trial')
    
        # # Correct order of experiment → MATLAB file
        # reordered_indices = [i - 1 for i in df['Experiment_id'].tolist()]
    
        # # Reorder
        # X = X[reordered_indices]                # (40 trials)
        # labels = labels[reordered_indices]
    
        # # Use only 32 EEG channels
        # X = X[:, :32, :]                         # (40, 32, 8064)
    
        # # Remove first 3 seconds (384 samples)
        # X = X[:, :, 128*3:]
    
        # # Binary labels
        # valence = (labels[:, 0] > 5).astype(int)
        # arousal = (labels[:, 1] > 5).astype(int)
    
        # # TRIAL indices for correct splitting
        # trial_indices = np.arange(40)
        # combined = valence * 2 + arousal
    
        # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    
        # for fold, (train_trials, test_trials) in enumerate(skf.split(np.zeros(len(combined)), combined)):
        #     print(f"\nFold {fold}")
        #     print(f"  Train trials: {train_trials}")
        #     print(f"  Test trials:  {test_trials}")
            
        #     # Print labels for each fold
        #     print("  Train valence labels:", valence[train_trials])
        #     print("  Train arousal labels:", arousal[train_trials])
        #     print("  Test valence labels: ", valence[test_trials])
        #     print("  Test arousal labels: ", arousal[test_trials])
        #     tr_x, tr_y, te_x, te_y = [], [], [], []
    
        #     # ----------------------
        #     # TRAIN TRIALS
        #     # ----------------------
        #     for i in train_trials:
        #         eeg = X[i]
        #         label = [valence[i], arousal[i]]
        #         segs, labs = segment_trial(eeg, label, window_size, stride)
        #         tr_x.extend(segs)
        #         tr_y.extend(labs)
    
        #     # ----------------------
        #     # TEST TRIALS
        #     # ----------------------
        #     for i in test_trials:
        #         eeg = X[i]
        #         label = [valence[i], arousal[i]]
    
        #         segs, labs = segment_trial(eeg, label, window_size, stride)
        #         te_x.extend(segs)
        #         te_y.extend(labs)
    
        #     # ----------------------
        #     # Convert to numpy arrays
        #     # ----------------------
        #     tr_x = np.array(tr_x)      # (N_train_segments, 32, window)
        #     te_x = np.array(te_x)
        #     tr_y = np.array(tr_y)      # (N_train_segments, 2)
        #     te_y = np.array(te_y)
    
        #     # ----------------------
        #     # Convert to torch tensors
        #     # ----------------------
        #     tr_x = torch.from_numpy(tr_x).float()
        #     te_x = torch.from_numpy(te_x).float()
        #     tr_y = torch.from_numpy(tr_y).long()
        #     te_y = torch.from_numpy(te_y).long()
    
        #     # ----------------------
        #     # SAVE
        #     # ----------------------
        #     save_path = os.path.join(save_dir, f"subject{sub}_fold{fold}.pkl")
        #     with open(save_path, "wb") as f:
        #         pickle.dump([tr_x, tr_y, te_x, te_y, train_trials, test_trials], f)
    
        for mode in modes:
            fold_accuracies = []
            fold_f1s=[]
            for fold in range(0,10):
                with open(f"D:\Deap\EEG\subject{sub}_fold{fold}_10fold.pkl", "rb") as f:
                    tr_x, tr_y, te_x, te_y, _, _ = pickle.load(f)
                
                # Then convert to PyTorch tensors
                tr_x = tr_x.float().unsqueeze(1)  # shape: (N_train, 1, chans, samples)
                te_x = te_x.float().unsqueeze(1)
                print(tr_x.shape)
                
                tr_val = tr_y[:, 0]
                tr_aro = tr_y[:, 1]
                te_val = te_y[:, 0]
                te_aro = te_y[:, 1]

            
            
                if mode=="aro":
                    tr_y = torch.from_numpy(np.array(tr_aro)).long()
                    te_y =torch.from_numpy(np.array(te_aro)).long()
                else:
                    tr_y = torch.from_numpy(np.array(tr_val)).long()
                    te_y =torch.from_numpy(np.array(te_val)).long()
         
        
                batch_size = 64
                

                # tr_y is already a torch tensor of shape [N]
                labels_np = tr_y.cpu().numpy()
                
                class_weights = compute_class_weight(
                    class_weight="balanced",
                    classes=np.array([0, 1]),
                    y=labels_np
                )
                
                class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

                
                train_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(TensorDataset(te_x, te_y), batch_size=batch_size, shuffle=False)
                
                #model_cnn= EEGNet_tor(nb_classes=2, Chans=32, Samples=512)
                model_cnn=ShallowConvNet(nb_classes=2, Chans=32, Samples=512, num_layers=4)
                
                model_cnn.to(device)
                
                
                
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                optimizer = optim.Adam(model_cnn.parameters(), lr=1e-4)
                # Training loop
                for epoch in range(400):
                    # ----- Training -----
                    model_cnn.train()
                    total_loss = 0
                    correct = 0
                
                    for batch_x, batch_y in train_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                        optimizer.zero_grad()
                        outputs = model_cnn(batch_x)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            if hasattr(model_cnn, 'fc'):
                                with torch.no_grad():
                                    model_cnn.fc.weight.data = torch.renorm(model_cnn.fc.weight.data, p=2, dim=0, maxnorm=0.5)
                            
                        total_loss += loss.item()
                        preds = outputs.argmax(dim=1)
                        correct += (preds == batch_y).sum().item()
                
                    train_acc = correct / len(train_loader.dataset)
                
                    # ----- Evaluation on Test Set -----
            
                    # ----- Evaluation on Test Set -----
                    model_cnn.eval()
                    test_correct = 0
                    
                    all_preds = []
                    all_labels = []
                    
                    with torch.no_grad():
                        for test_x, test_y in test_loader:
                            test_x, test_y = test_x.to(device), test_y.to(device)
                            test_outputs = model_cnn(test_x)
                    
                            test_preds = test_outputs.argmax(dim=1)
                    
                            test_correct += (test_preds == test_y).sum().item()
                    
                            all_preds.append(test_preds.cpu())
                            all_labels.append(test_y.cpu())
                    
                    # Accuracy
                    test_acc = test_correct / len(test_loader.dataset)
                    
                    # F1 score (macro is recommended for unbalanced EEG emotion data)
                    all_preds = torch.cat(all_preds).numpy()
                    all_labels = torch.cat(all_labels).numpy()
                    test_f1 = f1_score(all_labels, all_preds, average='macro')
                    
                                
                    print(
                        f"Subject {sub}, "
                        f"Fold {fold}, "
                        f"Epoch {epoch+1}, "
                        f"Test Acc: {test_acc:.4f}, "
                        f"Test F1: {test_f1:.4f}"
                    )

                    
                torch.save(model_cnn.state_dict(), f'D:\.spyder-py3\Finetuned_models_ratio7030\deap_shallowtrans_{sub}_{fold}fold_{mode}mode_10fold_classweight.pth')
                with open('eeg_cross_val_10folds.txt', 'a') as f:
                    f.write(
                        f"Subject {sub} for {mode} of fold {fold}: "
                        f"acc: {test_acc:.4f}, F1: {test_f1:.4f};\n"
                    )

                fold_accuracies.append(test_acc)
                fold_f1s.append(test_f1)

                
            avg_acc = sum(fold_accuracies) / len(fold_accuracies)
            std_acc = (sum((x - avg_acc)**2 for x in fold_accuracies) / len(fold_accuracies))**0.5
            avg_f1 = sum(fold_f1s) / len(fold_f1s)
            std_f1 = (sum((x - avg_f1)**2 for x in fold_f1s) / len(fold_f1s))**0.5
            
            print(f"Fold accuracies {mode}:", fold_accuracies)
            print(f"Average accuracy{mode}: {avg_acc:.4f}")
            print(f"Std accuracy: {std_acc:.4f}")
            print(f"Average F1 {mode}: {avg_f1:.4f}")
            print(f"Std F1: {std_f1:.4f}")
            with open('eeg_cross_val_average.txt', 'a') as f:
                f.write(f"sub{sub} for mode {mode}: avg acc: {avg_acc:.4f}; avg F1: {test_f1:.4f}\n")    
                
              
            
                
            
            # # #model = ShallowConvNet(nb_classes=2, Chans=32, Samples=256, num_layers=4)
            # # model = ViT_Encoder(cnn_model=model_cnn, classifier=True, img_size=[30,40], in_chans=1,
            # #                     patch_size=(30, 5), stride=[1, 5], depth=12, num_heads=1, embed_dim = 768,
            # #                     embed_eeg=True, embed_pos=False)
            
            # # model = ShallowConvNet(nb_classes=2, Chans=32, Samples=256, num_layers=4)
            
            # model_cnn= EEGNet_tor(nb_classes=2, Chans=32, Samples=256)
            # path= rf'D:\.spyder-py3\Finetuned_models_ratio7030\model_with_weights_eeg_finetuned_deap_valence_eegnet_{sub}.pth'
            # model_cnn.load_state_dict(torch.load(path))
            # for param in model_cnn.parameters():
            #     param.requires_grad = False
            # model = ViT_Encoder(cnn_model=model_cnn, classifier=True, img_size=[40, 30], in_chans=1,
            #                     patch_size=(30, 5), stride=[1, 5], depth=12, num_heads=1, embed_dim = 768,
            #                     embed_eeg=True, embed_pos=False)
            
            # # model_cnn = ShallowConvNet(nb_classes=2, Chans=32, Samples=256)
            
            # # model = ViT_Encoder(cnn_model=model_cnn, classifier=True, img_size=[30, 40], in_chans=1,
            # #                     patch_size=(30, 5), stride=[1, 5], depth=12, num_heads=1, embed_dim = 768,
            # #                     embed_eeg=True, embed_pos=False)
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # model.to(device)
            
            # criterion = nn.CrossEntropyLoss()
            # # optimizer = optim.Adam(model.parameters(), lr=0.00005)
            # optimizer = optim.Adam(model.parameters(), lr=0.001)
            # # Training loop
            # for epoch in range(400):
            #     # ----- Training -----
            #     model.train()
            #     total_loss = 0
            #     correct = 0
            
            #     for batch_x, batch_y in train_loader:
            #         batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            #         optimizer.zero_grad()
            #         outputs = model(batch_x)
            #         loss = criterion(outputs, batch_y)
            #         loss.backward()
            #         optimizer.step()
            #         with torch.no_grad():
            #             for name, param in model.named_parameters():
            #                 if 'weight' in name and param.dim() >= 2:
            #                     param.data = param.data.renorm_(p=2, dim=0, maxnorm=1.0)
            #         # with torch.no_grad():
            #         #     #model.conv2.weight.data = torch.renorm(model.conv2.weight.data, p=2, dim=0, maxnorm=2)
            #         #     model.fc.weight.data = torch.renorm(model.fc.weight.data, p=2, dim=0, maxnorm=0.5)
            #         # if hasattr(model, 'fc'):
            #         #     with torch.no_grad():
            #         #         model.fc.weight.data = torch.renorm(model.fc.weight.data, p=2, dim=0, maxnorm=0.5)
            #         # else:
            #         #     with torch.no_grad():
            #         #         model.eeg_embed.fc.weight.data = torch.renorm(model.eeg_embed.fc.weight.data, p=2, dim=0, maxnorm=0.5)
            #         total_loss += loss.item()
            #         preds = outputs.argmax(dim=1)
            #         correct += (preds == batch_y).sum().item()
            
            #     train_acc = correct / len(train_loader.dataset)
            
            #     # ----- Evaluation on Test Set -----
            #     model.eval()
            #     test_correct = 0
            #     with torch.no_grad():
            #         for test_x, test_y in test_loader:
            #             test_x, test_y = test_x.to(device), test_y.to(device)
            #             test_outputs = model(test_x)
            #             test_preds = test_outputs.argmax(dim=1)
            #             test_correct += (test_preds == test_y).sum().item()
            
            #     test_acc = test_correct / len(test_loader.dataset)
            
            #     print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
            # with open('eeg_results_new_25.txt', 'a') as f:
            #     f.write(f"Deap dataset Subject {sub} Testing Accuracy: {test_acc}\n")    
            
            #     if epoch == 400 - 1:
            #         torch.save(model.state_dict(), f'D:\.spyder-py3\Finetuned_models_ratio7030\model_with_weights_eeg_finetuned_deap_valence_eegnet_trans_12_{sub}.pth')
            
        
        # #####################################################################arousal
        
        # tr_y = torch.from_numpy(np.array(tr_y2)).long()
        # te_y =torch.from_numpy(np.array(te_y2)).long()
         
        
        # batch_size = 64
        
        # train_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=batch_size, shuffle=True)
        # test_loader = DataLoader(TensorDataset(te_x, te_y), batch_size=batch_size, shuffle=False)
        
        
        
        # model_cnn= EEGNet_tor(nb_classes=2, Chans=32, Samples=256)
        # path= rf'D:\.spyder-py3\Finetuned_models_ratio7030\model_with_weights_eeg_finetuned_deap_arousal_eegnet_{sub}.pth'
        # model_cnn.load_state_dict(torch.load(path))
        # for param in model_cnn.parameters():
        #     param.requires_grad = False
            
            
            
        # model = ViT_Encoder(cnn_model=model_cnn, classifier=True, img_size=[40, 30], in_chans=1,
        #                     patch_size=(30, 5), stride=[1, 5], depth=12, num_heads=1, embed_dim = 768,
        #                     embed_eeg=True, embed_pos=False)
       
        # # model = ViT_Encoder(cnn_model=model_cnn, classifier=True, img_size=[30, 40], in_chans=1,
        # #                     patch_size=(30, 5), stride=[1, 5], depth=12, num_heads=1, embed_dim = 768,
        # #                     embed_eeg=True, embed_pos=False)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device)
        
        # criterion = nn.CrossEntropyLoss()
        # # optimizer = optim.Adam(model.parameters(), lr=0.00005)
        # optimizer = optim.Adam(model.parameters(), lr=0.001)
        # # Training loop
        # for epoch in range(400):
        #     # ----- Training -----
        #     model.train()
        #     total_loss = 0
        #     correct = 0
        
        #     for batch_x, batch_y in train_loader:
        #         batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        #         optimizer.zero_grad()
        #         outputs = model(batch_x)
        #         loss = criterion(outputs, batch_y)
        #         loss.backward()
        #         optimizer.step()
        #         with torch.no_grad():
        #             for name, param in model.named_parameters():
        #                 if 'weight' in name and param.dim() >= 2:
        #                     param.data = param.data.renorm_(p=2, dim=0, maxnorm=1.0)
        #         # with torch.no_grad():
        #         #     #model.conv2.weight.data = torch.renorm(model.conv2.weight.data, p=2, dim=0, maxnorm=2)
        #         #     model.fc.weight.data = torch.renorm(model.fc.weight.data, p=2, dim=0, maxnorm=0.5)
        #         # if hasattr(model, 'fc'):
        #         #     with torch.no_grad():
        #         #         model.fc.weight.data = torch.renorm(model.fc.weight.data, p=2, dim=0, maxnorm=0.5)
        #         # else:
        #         #     with torch.no_grad():
        #         #         model.eeg_embed.fc.weight.data = torch.renorm(model.eeg_embed.fc.weight.data, p=2, dim=0, maxnorm=0.5)
        #         total_loss += loss.item()
        #         preds = outputs.argmax(dim=1)
        #         correct += (preds == batch_y).sum().item()
        
        #     train_acc = correct / len(train_loader.dataset)
        
        #     # ----- Evaluation on Test Set -----
        #     model.eval()
        #     test_correct = 0
        #     with torch.no_grad():
        #         for test_x, test_y in test_loader:
        #             test_x, test_y = test_x.to(device), test_y.to(device)
        #             test_outputs = model(test_x)
        #             test_preds = test_outputs.argmax(dim=1)
        #             test_correct += (test_preds == test_y).sum().item()
        
        #     test_acc = test_correct / len(test_loader.dataset)
        
           
        #     print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
            
        #     if epoch == 400 - 1:
        #         with open('eeg_results_new_25.txt', 'a') as f:
        #             f.write(f"Deap dataset Subject {sub} Arousal Testing Accuracy: {test_acc}\n")
        #     #     torch.save(model.state_dict(), f'D:\.spyder-py3\Finetuned_models_ratio7030\model_with_weights_eeg_finetuned_deap_arousal_eegnet_trans_12_{sub}.pth')
        
