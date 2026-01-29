import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader
from transformers import ASTFeatureExtractor
import torch.nn.functional as F
from timm.layers import Mlp, DropPath, use_fused_attn
from Dataload_audio import DataLoadAudio
from EAV_datasplit import EAVDataSplit
import numpy as np


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
        self.dense = nn.Linear(F2 * 65, 5)
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
class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
class Attention(nn.Module):
    #fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,  # should be true
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        #q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,

            #init_values: Optional[float] = None,
            init_values=None,  # mhlee
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
class EEG_decoder(nn.Module):
    def __init__(self, eeg_channel = 30, dropout=0.1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                eeg_channel, eeg_channel, 11, 1, padding=5, bias=False
            ),
            nn.BatchNorm1d(eeg_channel),
            nn.ReLU(True),
            nn.Dropout1d(dropout),
            nn.Conv1d(
                eeg_channel, eeg_channel * 2, 11, 1, padding=5, bias=False
            ),
            #nn.BatchNorm1d(eeg_channel * 2),
        )
    def forward(self, x):
        x = self.conv(x)
        return x
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    dynamic_img_pad: torch.jit.Final[bool]
    def __init__(
            self,
            img_size = [40, 65],
            patch_size = [40, 5],
            in_chans: int = 1,
            stride = [1, 5],
            embed_dim: int = 10,
            norm_layer = None,
            flatten: bool = True,
            output_fmt = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = tuple(patch_size)

        if img_size is not None:
            self.img_size = tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

            # flatten spatial dim and transpose to channels last, kept for bwd compat
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        # updated_mh
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, bias=bias)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        #print(x.shape)
        x=x.permute(0,1,3,2) # needs to be permuted according to the outputs from the ast
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x
class ViT_Encoder(nn.Module):
    def __init__(self, cnn_model = False, img_size=[40, 227], in_chans = 1, patch_size=16, stride = 16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 classifier : bool = False, num_classes = 5, embed_eeg = True, embed_pos = False):
        super().__init__()
        # updated_mh
        #self.num_patches = (img_size // patch_size) ** 2
        self.embed_eeg = embed_eeg
        self.embed_pos = embed_pos
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.dropout = nn.Dropout(0.5)

        #num_patches_height = (img_size[0] - patch_size[0]) // stride + 1
        #num_patches_width = (img_size[1] - patch_size[1]) // stride + 1
        #self.total_patches = num_patches_height * num_patches_width

        self.stride = stride
        if embed_eeg:
            if cnn_model:  # pretrained CNN model
                self.eeg_embed = cnn_model
            else:
                self.eeg_embed = EEGNet_tor(nb_classes=5, Chans=30, Samples=500)
        self.patch_embed = PatchEmbed(img_size = img_size, patch_size  = patch_size, stride = stride, embed_dim = embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.total_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)

        self.feature_map = None  # this will contain the ViT feature map (including CLASS token)

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        if classifier:
            self.head = nn.Linear(embed_dim, num_classes, bias=True)
        else:
            self.head = []
    def feature(self, x): # Returns the transformer feature map for all input data, bypassing the classifier head.
        #print(x.shape)
        B = x.shape[0]
        if self.embed_eeg:  # Only for EEG
            x = self.eeg_embed.get_feature_map(x)
            x = self.dropout(x)  # 추가적으로넣어줌
            x = x.squeeze(2)
            x = x.unsqueeze(1)
            
        #print(x.shape) 
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Copy
        x = torch.cat((cls_tokens, x), dim=1)  # Add class token

        if self.embed_pos:
            x += self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) # Return the feature map including the class token
        return x

    def forward(self, x):
        #print(x.shape)
        B = x.shape[0]
        if self.embed_eeg: ## only for the EEG
            x = self.eeg_embed.get_feature_map(x)
            x = self.dropout(x)  # 추가적으로넣어줌
            x = x.squeeze(2)
            x = x.unsqueeze(1)
        
        #print(x.shape)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # 복제
        x = torch.cat((cls_tokens, x), dim=1)  # 클래스 토큰 추가

        if self.embed_pos:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        self.feature_map = x

        if self.head:  # classifier mode
            x = self.head(x[:, 0])
        return x
    
    
class Trainer_uni:
    def __init__(self, model, data, lr=1e-3, batch_size=32, num_epochs=10, device=None,sub=0):

        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.sub=sub
        self.tr_x, self.tr_y, self.te_x, self.te_y = data
        self.train_dataloader = self._prepare_dataloader(self.tr_x, self.tr_y, shuffle=True)
        self.test_dataloader = self._prepare_dataloader(self.te_x, self.te_y, shuffle=False)

        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        #self.device = torch.device("cpu")  
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def _prepare_dataloader(self, x, y, shuffle=False):
        dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def train(self):
        self.model.train()  # Set model to training mode
        for epoch in range(self.num_epochs):
            for batch_idx, (data, targets) in enumerate(self.train_dataloader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                scores = self.model(data)
                loss = self.criterion(scores, targets)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                
                # if hasattr(self.model, 'module'):
                #     with torch.no_grad():
                #         for name, param in model.module.named_parameters():
                #             if 'weight' in name and param.dim() >= 2:
                #                 param.data = param.data.renorm_(p=2, dim=0, maxnorm=1.0)
                # else:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if 'weight' in name and param.dim() >= 2:
                            param.data = param.data.renorm_(p=2, dim=0, maxnorm=1.0)


                if batch_idx % 100 == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Step [{batch_idx}/{len(self.train_dataloader)}], Loss: {loss.item():.4f}")

            if self.test_dataloader:
                acc=self.validate()
                
            if (epoch==self.num_epochs-1):    
                save_path = os.path.join(r'D:\.spyder-py3\Finetuned_models_ratio7030', f'eeg_finetuned_eegnet_transf350_{self.sub}.pth')
                torch.save(self.model.state_dict(), save_path)
                with open('eeg_eegnet_transf.txt', 'a') as f:
                    f.write(f"Subject {self.sub} Epoch {epoch + 1} Testing Accuracy: {acc}\n")

    def validate(self):
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0
        total_correct = 0
        with torch.no_grad():
            for data, targets in self.test_dataloader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                scores = self.model(data)
                loss = self.criterion(scores, targets)
                total_loss += loss.item()
                predictions = scores.argmax(dim=1)
                total_correct += (predictions == targets).sum().item()

        avg_loss = total_loss / len(self.test_dataloader)
        accuracy = total_correct / len(self.test_dataloader.dataset)
        print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return accuracy
if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import os, pickle
    i=2
    for i in range(1, 43):
        file_path = r"C:\Users\user.DESKTOP-HI4HHBR\Downloads\EEG\EEG"
        file_name = f"subject_{i:02d}_eeg.pkl"
        file_ = os.path.join(file_path, file_name)
        # You can directly work from here
        with open(file_, 'rb') as f:
            eeg_list = pickle.load(f)
        tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = eeg_list
        tr_x_eeg = torch.from_numpy(tr_x_eeg).float().unsqueeze(1).to(device)
        te_x_eeg = torch.from_numpy(te_x_eeg).float().unsqueeze(1).to(device)
        data = [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg]
        
        #model(te_x_eeg)
        
        model_cnn = EEGNet_tor(nb_classes=5, Chans=30, Samples=500)
        path= os.path.join(r'D:\.spyder-py3\finetuned_cnn_7030(1)', f'eeg_finetuned_eegnet_{i}.pth')
        model_cnn.load_state_dict(torch.load(path))
        
        #aa = model_cnn(te_x_eeg)
        for param in model_cnn.parameters():
            param.requires_grad = False
        
        model = ViT_Encoder(cnn_model=model_cnn, classifier=True, img_size=[40, 65], in_chans=1,
                            patch_size=(40, 5), stride=[1, 5], depth=12, num_heads=1, embed_dim = 768,
                            embed_eeg=True, embed_pos=False)
        
        trainer = Trainer_uni(model=model, data=data, lr=1e-5, batch_size=32, num_epochs=30,sub=i)
        trainer.train()
        
        for param in model_cnn.parameters():
            param.requires_grad = True
        
        trainer = Trainer_uni(model=model, data=data, lr=1e-5, batch_size=32, num_epochs=50,sub=i)
        trainer.train()
    
    
    #out = model_cnn(te_x_eeg)
    #out = model_cnn.get_feature_map(te_x_eeg)