import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ============================================================
# Basic Operators & Attention Modules
# ============================================================

class NormDownsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale=0.5, use_norm=False):
        super(NormDownsample, self).__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.GroupNorm(1, out_ch) 
        self.prelu = nn.PReLU()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=False),
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
        )
        
    def forward(self, x):
        x = self.prelu(self.down(x))
        return self.norm(x) if self.use_norm else x

class NormUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2, use_norm=False):
        super(NormUpsample, self).__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.GroupNorm(1, out_ch)
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias=False),
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
        )
        self.up = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
            
    def forward(self, x, y):
        x = self.up_scale(x)
        if x.shape[2:] != y.shape[2:]:
            x = F.interpolate(x, size=(y.shape[2], y.shape[3]), mode='bilinear', align_corners=False)
        x = self.prelu(self.up(torch.cat([x, y], dim=1)))
        return self.norm(x) if self.use_norm else x

class CAB(nn.Module):
    """ Cross Attention Block """
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.log_temperature = nn.Parameter(torch.zeros(num_heads, 1, 1))
        
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, padding_mode='replicate', bias=bias)
        
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, padding_mode='replicate', bias=bias)
        
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        k, v = self.kv_dwconv(self.kv(y)).chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        temp = F.softplus(self.log_temperature) + 1e-6
        attn = (q @ k.transpose(-2, -1)) * temp
        attn = F.softmax(attn, dim=-1)

        out = rearrange(attn @ v, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)

class IEL(nn.Module):
    """ Intensity Enhancement Layer (SimpleGate) """
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        hidden_features = (hidden_features + 3) // 4 * 4 

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, padding_mode='replicate', bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = x1 * x2
        return self.project_out(x)

class LCABlock(nn.Module):
    """ Lightweight Cross Attention Block """
    def __init__(self, dim, num_heads, bias=False):
        super(LCABlock, self).__init__()
        self.norm_x = nn.GroupNorm(1, dim)
        self.norm_y = nn.GroupNorm(1, dim)
        self.norm_ffn = nn.GroupNorm(1, dim)
        
        self.ffn = CAB(dim, num_heads, bias=bias)
        self.gdfn = IEL(dim) 
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm_x(x), self.norm_y(y))
        x = x + self.gdfn(self.norm_ffn(x))
        return x

class ChromaConfidenceGate(nn.Module):
    def __init__(self, alpha=0.05, gamma_min=1.0, gamma_max=5.0):
        super().__init__()
        self.alpha = alpha
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.gamma_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, i_in):
        i = i_in.clamp(0.0, 1.0)
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(self.gamma_logit)

        base = torch.sin(0.5 * math.pi * i).clamp_min(1e-4)
        gate = base.pow(gamma)
        gate = self.alpha + (1.0 - self.alpha) * gate
        return gate

# ============================================================
# Single-Stage ICD Network
# ============================================================

class ICD(nn.Module):
    def __init__(self, channels=[24, 32, 64, 96], heads=[1, 2, 4, 4], norm=False, use_gate=True):
        super(ICD, self).__init__()
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        self.unshuffle = nn.PixelUnshuffle(2)
        self.shuffle = nn.PixelShuffle(2)

        # --- C Stream (Chrominance Encoder/Decoder) ---
        self.CE_block0 = nn.Conv2d(12, ch1, 3, 1, 1, padding_mode='replicate', bias=False)
        self.CE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.CE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.CE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        
        self.CD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.CD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.CD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.CD_block0 = nn.Conv2d(ch1, 12, 3, 1, 1, padding_mode='replicate', bias=False)
        
        # --- I Stream (Intensity Encoder/Decoder) ---
        self.IE_block0 = nn.Conv2d(4, ch1, 3, 1, 1, padding_mode='replicate', bias=False)
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Conv2d(ch1, 4, 3, 1, 1, padding_mode='replicate', bias=False)
        
        # --- Cross Attention Network ---
        self.C_LCA1 = LCABlock(ch2, head2)
        self.C_LCA2 = LCABlock(ch3, head3)
        self.C_LCA3 = LCABlock(ch4, head4)
        self.C_LCA4 = LCABlock(ch4, head4)
        self.C_LCA5 = LCABlock(ch3, head3)
        self.C_LCA6 = LCABlock(ch2, head2)
        
        self.I_LCA1 = LCABlock(ch2, head2)
        self.I_LCA2 = LCABlock(ch3, head3)
        self.I_LCA3 = LCABlock(ch4, head4)
        self.I_LCA4 = LCABlock(ch4, head4)
        self.I_LCA5 = LCABlock(ch3, head3)
        self.I_LCA6 = LCABlock(ch2, head2)
        
        self.use_gate = use_gate
        if self.use_gate:
            self.chroma_gate = ChromaConfidenceGate()

    def forward(self, x):
        eps = 1e-4
        b, c, h, w = x.shape
        
        # 1. Dynamic Padding (ensure spatial dims are multiples of 16)
        base_pad = 8
        pad_h_extra = (16 - (h + base_pad * 2) % 16) % 16
        pad_w_extra = (16 - (w + base_pad * 2) % 16) % 16
        x_pad = F.pad(x, (base_pad, base_pad + pad_w_extra, base_pad, base_pad + pad_h_extra), mode='reflect')

        # 2. Extract Physical Priors
        i_in = torch.max(x_pad, dim=1, keepdim=True)[0].clamp_min(eps)
        c_raw = torch.log(x_pad.clamp_min(eps)) - torch.log(i_in)

        if self.use_gate:
            c_gate = self.chroma_gate(i_in)
            c_in = c_gate * c_raw  
        else:
            c_in = c_raw        

        # 3. Spatial Unshuffle
        i_in_fold = self.unshuffle(i_in)
        c_in_fold = self.unshuffle(c_in)

        # 4. Dual-Stream LCA Backbone
        i_enc0, c_0 = self.IE_block0(i_in_fold), self.CE_block0(c_in_fold)
        i_enc1, c_1 = self.IE_block1(i_enc0), self.CE_block1(c_0)
        
        i_lca1 = self.I_LCA1(i_enc1, c_1)
        c_lca1 = self.C_LCA1(c_1, i_enc1)
        i_enc2, c_2 = self.IE_block2(i_lca1), self.CE_block2(c_lca1)

        i_lca2 = self.I_LCA2(i_enc2, c_2)
        c_lca2 = self.C_LCA2(c_2, i_enc2)
        i_enc3, c_3 = self.IE_block3(i_lca2), self.CE_block3(c_lca2)

        i_lca3 = self.I_LCA3(i_enc3, c_3)
        c_lca3 = self.C_LCA3(c_3, i_enc3)

        i_lca4 = self.I_LCA4(i_lca3, c_lca3)
        c_lca4 = self.C_LCA4(c_lca3, i_lca3)

        c_dec3 = self.CD_block3(c_lca4, c_lca2)
        i_dec3 = self.ID_block3(i_lca4, i_lca2)

        i_lca5 = self.I_LCA5(i_dec3, c_dec3)
        c_lca5 = self.C_LCA5(c_dec3, i_dec3)

        c_dec2 = self.CD_block2(c_lca5, c_lca1)
        i_dec2 = self.ID_block2(i_lca5, i_lca1)

        i_lca6 = self.I_LCA6(i_dec2, c_dec2)
        c_lca6 = self.C_LCA6(c_dec2, i_dec2)

        c_dec1 = self.CD_block1(c_lca6, c_0)
        i_dec1 = self.ID_block1(i_lca6, i_enc0)

        # Residual Prediction
        delta_i_fold = self.ID_block0(i_dec1)
        delta_c_fold = self.CD_block0(c_dec1)

        # 5. Spatial Shuffle & Physical Reconstruction
        delta_i = self.shuffle(delta_i_fold)
        delta_c = self.shuffle(delta_c_fold)
        
        i_out = torch.clamp(i_in + delta_i, min=eps, max=1.0)
        lower_bound = math.log(eps) 
        c_out = torch.clamp(c_in + delta_c, min=lower_bound, max=0.0)

        # Convert back to RGB space
        output_rgb = torch.clamp(torch.exp(c_out) * i_out, 0.0, 1.0)
        
        # Remove padding
        output_rgb = output_rgb[:, :, base_pad : base_pad + h, base_pad : base_pad + w]
        i_out = i_out[:, :, base_pad : base_pad + h, base_pad : base_pad + w]
        c_out = c_out[:, :, base_pad : base_pad + h, base_pad : base_pad + w]
        
        return output_rgb, i_out, c_out