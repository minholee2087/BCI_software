import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def batch_rodrigues(rot_vecs, epsilon=1e-8):
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype
    
    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)
    
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view((batch_size, 3, 3))
    
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    
    return rot_mat


class FLAMEHead(nn.Module):
    def __init__(self, n_shape=100, n_exp=50, resolution=64):
        super(FLAMEHead, self).__init__()
        
        self.n_shape = n_shape
        self.n_exp = n_exp
        self.resolution = resolution
        
        vertices, faces = self._create_head_mesh(resolution)
        
        self.register_buffer('v_template', torch.from_numpy(vertices).float())
        self.register_buffer('faces', torch.from_numpy(faces).long())
        
        n_verts = vertices.shape[0]
        
        shapedirs = self._create_shape_basis(n_verts, n_shape)
        expdirs = self._create_expression_basis(n_verts, n_exp, vertices)
        
        self.register_buffer('shapedirs', torch.from_numpy(shapedirs).float())
        self.register_buffer('expdirs', torch.from_numpy(expdirs).float())
        
    def _create_head_mesh(self, resolution):
        n_phi = resolution
        n_theta = resolution * 2
        
        phi = np.linspace(0, np.pi, n_phi)
        theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
        
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        phi_flat = phi_grid.flatten()
        theta_flat = theta_grid.flatten()
        
        r = 0.1
        
        x_base = r * np.sin(phi_flat) * np.cos(theta_flat)
        y_base = r * np.sin(phi_flat) * np.sin(theta_flat)
        z_base = r * np.cos(phi_flat)
        
        y_base *= 1.3
        z_base *= 0.95
        
        face_front = (theta_flat > np.pi/2) & (theta_flat < 3*np.pi/2)
        face_region = face_front & (phi_flat > np.pi/4) & (phi_flat < 3*np.pi/4)
        
        face_scale = np.where(face_region, 1.0 + 0.15 * np.sin((phi_flat - np.pi/4) * 2), 1.0)
        x_base *= face_scale
        
        chin_region = face_front & (phi_flat > 2*np.pi/3)
        chin_depth = np.where(chin_region, 
                             0.015 * np.sin((phi_flat - 2*np.pi/3) * 3) * np.exp(-((theta_flat - np.pi)**2) / 0.5),
                             0)
        x_base += chin_depth
        
        forehead_region = face_front & (phi_flat < np.pi/3)
        forehead_curve = np.where(forehead_region,
                                  0.01 * np.cos(phi_flat * 3),
                                  0)
        x_base += forehead_curve
        
        nose_region = face_front & (np.abs(theta_flat - np.pi) < 0.3) & (phi_flat > np.pi/3) & (phi_flat < 2*np.pi/3)
        nose_protrusion = np.where(nose_region,
                                   0.02 * np.exp(-((theta_flat - np.pi)**2) / 0.1) * np.sin((phi_flat - np.pi/3) * 1.5),
                                   0)
        x_base += nose_protrusion
        
        vertices = np.stack([x_base, y_base, z_base], axis=1).astype(np.float32)
        
        faces = []
        for i in range(n_theta):
            for j in range(n_phi - 1):
                v0 = i * n_phi + j
                v1 = i * n_phi + j + 1
                v2 = ((i + 1) % n_theta) * n_phi + j
                v3 = ((i + 1) % n_theta) * n_phi + j + 1
                
                faces.append([v0, v2, v1])
                faces.append([v1, v2, v3])
                
        faces = np.array(faces, dtype=np.int64)
        
        return vertices, faces
    
    def _create_shape_basis(self, n_verts, n_components):
        shapedirs = np.zeros((n_verts, 3, n_components), dtype=np.float32)
        
        coords = self.v_template.numpy() if hasattr(self, 'v_template') else np.zeros((n_verts, 3))
        
        for i in range(n_components):
            freq = 1 + i // 10
            phase = i * 0.5
            
            if i % 5 == 0:
                shapedirs[:, 0, i] = 0.005 * np.sin(freq * coords[:, 1] + phase)
            elif i % 5 == 1:
                shapedirs[:, 1, i] = 0.005 * np.sin(freq * coords[:, 2] + phase)
            elif i % 5 == 2:
                shapedirs[:, 2, i] = 0.005 * np.cos(freq * coords[:, 0] + phase)
            elif i % 5 == 3:
                shapedirs[:, 0, i] = 0.003 * np.sin(freq * coords[:, 0] * coords[:, 1] + phase)
                shapedirs[:, 1, i] = 0.003 * np.cos(freq * coords[:, 0] * coords[:, 1] + phase)
            else:
                shapedirs[:, :, i] = 0.002 * np.random.randn(n_verts, 3)
                
        return shapedirs
    
    def _create_expression_basis(self, n_verts, n_components, vertices):
        expdirs = np.zeros((n_verts, 3, n_components), dtype=np.float32)
        
        theta = np.arctan2(vertices[:, 1], vertices[:, 0])
        phi = np.arccos(np.clip(vertices[:, 2] / (np.linalg.norm(vertices, axis=1) + 1e-8), -1, 1))
        
        face_front = (theta > np.pi/2) & (theta < 3*np.pi/2)
        lower_face = phi > np.pi/2
        upper_face = phi < np.pi/2
        
        mouth_region = face_front & (np.abs(theta - np.pi) < 0.4) & (phi > 0.55*np.pi) & (phi < 0.75*np.pi)
        eye_region_left = face_front & (theta > 0.7*np.pi) & (theta < 0.9*np.pi) & (phi > 0.35*np.pi) & (phi < 0.5*np.pi)
        eye_region_right = face_front & (theta > 1.1*np.pi) & (theta < 1.3*np.pi) & (phi > 0.35*np.pi) & (phi < 0.5*np.pi)
        brow_region = face_front & (phi > 0.25*np.pi) & (phi < 0.4*np.pi)
        cheek_region = face_front & (phi > 0.45*np.pi) & (phi < 0.65*np.pi) & ((theta < 0.8*np.pi) | (theta > 1.2*np.pi))
        nose_region = face_front & (np.abs(theta - np.pi) < 0.2) & (phi > 0.4*np.pi) & (phi < 0.6*np.pi)
        
        if n_components > 0:
            expdirs[mouth_region, 0, 0] = 0.015
            expdirs[cheek_region, 0, 0] = 0.008
            
        if n_components > 1:
            expdirs[mouth_region, 2, 1] = -0.012
            
        if n_components > 2:
            expdirs[mouth_region, 0, 2] = -0.01
            expdirs[brow_region, 0, 2] = -0.005
            
        if n_components > 3:
            expdirs[brow_region, 2, 3] = 0.008
            expdirs[eye_region_left | eye_region_right, 2, 3] = 0.005
            
        if n_components > 4:
            expdirs[brow_region, 2, 4] = -0.006
            expdirs[eye_region_left | eye_region_right, 2, 4] = -0.004
            
        if n_components > 5:
            left_cheek = cheek_region & (theta < np.pi)
            right_cheek = cheek_region & (theta > np.pi)
            expdirs[left_cheek, 0, 5] = 0.01
            expdirs[right_cheek, 0, 5] = 0.01
            expdirs[left_cheek, 1, 5] = -0.005
            expdirs[right_cheek, 1, 5] = 0.005
            
        if n_components > 6:
            upper_lip = mouth_region & (phi < 0.65*np.pi)
            lower_lip = mouth_region & (phi > 0.65*np.pi)
            expdirs[upper_lip, 2, 6] = -0.008
            expdirs[lower_lip, 2, 6] = 0.008
            
        if n_components > 7:
            expdirs[nose_region, 0, 7] = 0.005
            expdirs[nose_region, 1, 7] = 0.003 * np.sign(theta[nose_region] - np.pi)
            
        for i in range(8, n_components):
            freq = 1 + (i - 8) // 5
            region_idx = i % 5
            
            if region_idx == 0:
                expdirs[mouth_region, 0, i] = 0.005 * np.sin(freq * phi[mouth_region])
            elif region_idx == 1:
                expdirs[eye_region_left | eye_region_right, 2, i] = 0.003 * np.cos(freq * theta[eye_region_left | eye_region_right])
            elif region_idx == 2:
                expdirs[brow_region, 2, i] = 0.004 * np.sin(freq * theta[brow_region])
            elif region_idx == 3:
                expdirs[cheek_region, 0, i] = 0.004 * np.cos(freq * phi[cheek_region])
            else:
                random_region = np.random.rand(n_verts) > 0.7
                expdirs[random_region & face_front, :, i] = 0.002 * np.random.randn(np.sum(random_region & face_front), 3)
                
        return expdirs
    
    def forward(self, shape_params=None, expression_params=None, pose_params=None):
        batch_size = 1
        
        if shape_params is not None:
            batch_size = shape_params.shape[0]
        elif expression_params is not None:
            batch_size = expression_params.shape[0]
        elif pose_params is not None:
            batch_size = pose_params.shape[0]
            
        v = self.v_template.unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        if shape_params is not None:
            n_shape = min(shape_params.shape[1], self.n_shape)
            shape_offsets = torch.einsum('bl,mkl->bmk', shape_params[:, :n_shape], self.shapedirs[:, :, :n_shape])
            v = v + shape_offsets
            
        if expression_params is not None:
            n_exp = min(expression_params.shape[1], self.n_exp)
            exp_offsets = torch.einsum('bl,mkl->bmk', expression_params[:, :n_exp], self.expdirs[:, :, :n_exp])
            v = v + exp_offsets
            
        if pose_params is not None and pose_params.shape[1] >= 3:
            rot_mats = batch_rodrigues(pose_params[:, :3])
            v = torch.bmm(v, rot_mats.transpose(1, 2))
            
            if pose_params.shape[1] >= 6:
                translation = pose_params[:, 3:6].unsqueeze(1)
                v = v + translation
                
        return v, self.faces


class EEGToFLAMEModel(nn.Module):
    def __init__(self, eeg_channels=30, eeg_seq_len=7, n_shape=100, n_exp=50, hidden_dim=128):
        super(EEGToFLAMEModel, self).__init__()
        
        self.n_exp = n_exp
        self.n_pose = 6
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(eeg_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.expression_head = nn.Linear(hidden_dim, n_exp)
        self.pose_head = nn.Linear(hidden_dim, self.n_pose)
        
        self.flame = FLAMEHead(n_shape=n_shape, n_exp=n_exp)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        nn.init.zeros_(self.expression_head.weight)
        nn.init.zeros_(self.expression_head.bias)
        nn.init.zeros_(self.pose_head.weight)
        nn.init.zeros_(self.pose_head.bias)
        
    def forward(self, eeg_data, shape_params=None):
        x = eeg_data.transpose(1, 2)
        
        x = self.conv_layers(x)
        
        x = x.transpose(1, 2)
        
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        
        x = x.mean(dim=1)
        
        x = self.fc_layers(x)
        
        expression_params = self.expression_head(x)
        pose_params = self.pose_head(x)
        
        pose_params = pose_params * 0.1
        
        vertices, faces = self.flame(
            shape_params=shape_params,
            expression_params=expression_params,
            pose_params=pose_params
        )
        
        return vertices, faces, expression_params, pose_params
    
    def get_flame_params(self, eeg_data):
        with torch.no_grad():
            x = eeg_data.transpose(1, 2)
            x = self.conv_layers(x)
            x = x.transpose(1, 2)
            attn_out, _ = self.attention(x, x, x)
            x = self.norm(x + attn_out)
            x = x.mean(dim=1)
            x = self.fc_layers(x)
            
            expression_params = self.expression_head(x)
            pose_params = self.pose_head(x) * 0.1
            
            return expression_params, pose_params
