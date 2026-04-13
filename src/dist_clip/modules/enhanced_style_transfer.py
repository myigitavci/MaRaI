import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

def window_partition(x, window_size):
    """
    x: [B, C, H, W]
    returns windows: [B*num_windows, C, Wh, Ww], and (H, W)
    """
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))  # pad (left, right, top, bottom)

    H_pad, W_pad = x.shape[2], x.shape[3]
    x = x.view(B, C, H_pad // window_size, window_size, W_pad // window_size, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return x, (H_pad, W_pad)

def window_reverse(windows, window_size, H_orig, W_orig):
    B_num = windows.shape[0] // ((-(-H_orig // window_size) * -(-W_orig // window_size)))
    H_pad = ((H_orig + window_size - 1) // window_size) * window_size
    W_pad = ((W_orig + window_size - 1) // window_size) * window_size
    x = windows.view(B_num, H_pad // window_size, W_pad // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B_num, -1, H_pad, W_pad)
    return x[:, :, :H_orig, :W_orig]

class EnhancedStyleTransferV2(nn.Module):
    """
    Sharpness-oriented text-to-style module:
      - Global attention at low res; windowed attention at high res.
      - Spatial (per-pixel) gamma/beta modulation (SPADE/FILM-like).
    """
    def __init__(
        self,
        style_dim=512,
        hidden_dim=256,
        num_heads=8,
        window_size=8,
        norm_mode="bottleneck_only",  # 'always' | 'never' | 'bottleneck_only'
        use_dropout=True,  # Set False to match old checkpoints without dropout
    ):
        super().__init__()
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.norm_mode = norm_mode

        # Style encoder (text -> hidden)
        layers = [
            nn.Linear(style_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.1))
        layers.extend([
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        ])
        self.style_encoder = nn.Sequential(*layers)

        # Multi-head attention in hidden_dim
        # Multi-head attention in hidden_dim
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # Pre-register qkv projection so it's present during state_dict loading
        # Matches the windowed-attention branch input/output sizes (hidden_dim -> 3*hidden_dim)
        self.qkv_linear = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)        
        # Content adaptor when channels != hidden_dim
        self.in_adapt = nn.Conv2d(hidden_dim, hidden_dim, 1)  # no-op if already hidden_dim
        # We’ll create per-C adaptors lazily like you did, but also keep a default conv.

        # Spatial γ/β prediction head:
        # Combine normalized content + broadcast style map -> convs -> 2*C maps
        self.style_to_map = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.modulator = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        )

        # LayerNorm on channels (applied conditionally)
        self.ln = nn.LayerNorm(hidden_dim)

        # Strength scalars (tunable)
        self.gamma_scale = 5.0
        self.beta_scale  = 5.0

    def _maybe_adapt_in(self, x):
        # x: [B, C, H, W]; adapt to hidden_dim if needed
        B, C, H, W = x.shape
        if C == self.hidden_dim:
            return x
        key = f"_adapt_{C}->{self.hidden_dim}"
        if not hasattr(self, key):
            setattr(self, key, nn.Conv2d(C, self.hidden_dim, 1))
        return getattr(self, key)(x)

    def _apply_attention(self, x, style_vec):
        """
        x: [B, C(hidden), H, W]
        style_vec: [B, hidden]
        Global attention if small; shifted windowed attention if large.
        Returns: [B, C, H, W]
        """
        B, C, H, W = x.shape
        use_window = (H >= self.window_size * 2) and (W >= self.window_size * 2)

        if use_window:
            # --- Linear attention over all pixels ---
            device = x.device
            
            # Reshape x to [B, H*W, C] for linear attention
            B, C, H, W = x.shape
            x_flat = x.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
            
            # qkv projection (ensure on correct device)
            if not hasattr(self, 'qkv_linear'):
                self.qkv_linear = nn.Linear(C, C*3, bias=False).to(device)
            qkv = self.qkv_linear(x_flat)  # [B, H*W, 3*C]
            qkv = qkv.view(B, H*W, 3, C).permute(2, 0, 1, 3)  # [3, B, H*W, C]
            q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, H*W, C]

            # Linear attention computation with better numerical stability
            # Apply softmax to q and k for linear attention
            q = F.softmax(q, dim=-1)  # softmax over feature dimension
            k = F.softmax(k, dim=1)   # softmax over sequence dimension
            
            # Add small epsilon to prevent numerical issues
            eps = 1e-8
            q = q + eps
            k = k + eps
            
            # Compute context: K^T * V (more efficient than einsum for large batches)
            context = torch.matmul(k.transpose(1, 2), v)  # [B, C, C]
            
            # Compute output: Q * context
            out = torch.matmul(q, context)  # [B, H*W, C]
            
            # Reshape back to [B, C, H, W]
            out = out.permute(0, 2, 1).view(B, C, H, W)
            
            # Apply output projection if needed
            if hasattr(self, 'proj'):
                out = self.proj(out)
            
            return out
        else:
            # global attention over all spatial tokens
            L = H * W
            x_flat = x.view(B, C, L).permute(0, 2, 1)                     # [B, L, C]
            style_q = style_vec.unsqueeze(1).expand(B, L, C)              # [B, L, C]
            out, _ = self.attn(query=style_q, key=x_flat, value=x_flat)   # [B, L, C]
            return out.permute(0, 2, 1).view(B, C, H, W)

    def _maybe_norm(self, x, is_bottleneck):
        # x: [B, C, H, W]
        if self.norm_mode == "never":
            return x
        if self.norm_mode == "bottleneck_only" and not is_bottleneck:
            return x
        # Channel-wise LayerNorm over last dim requires permute
        B, C, H, W = x.shape
        x_ = x.permute(0, 2, 3, 1)                  # [B, H, W, C]
        x_ = self.ln(x_)                            # LN over C
        return x_.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]

    def forward(self, x, style_text_feat, use_attention=True, is_bottleneck=False):
        """
        x: [B, C, H, W]
        style_text_feat: [B, style_dim]
        """
        B, C, H, W = x.shape

        # 1) Adapt channels to hidden_dim
        x = self._maybe_adapt_in(x)  # [B, hidden, H, W]

        # 2) Encode style to hidden
        s = self.style_encoder(style_text_feat)  # [B, hidden]

        # 3) Attention (global or windowed depending on size)
        if use_attention:
            x_att = self._apply_attention(x, s)  # [B, hidden, H, W]
        else:
            x_att = x

        # 4) Conditional normalization (only where helpful)
        x_normed = self._maybe_norm(x_att, is_bottleneck=is_bottleneck)  # [B, hidden, H, W]

        # 5) Spatial γ/β prediction and modulation
        # Broadcast style to spatial map and fuse with content
        B, C, Hx, Wx = x_normed.shape
        s_map = self.style_to_map(s).unsqueeze(-1).unsqueeze(-1).expand(B, self.hidden_dim, Hx, Wx)
        fusion = torch.cat([x_normed, s_map], dim=1)                    # [B, 2C, H, W]
        gb = self.modulator(fusion)                                       # [B, 2C, H, W]
        gamma, beta = gb.chunk(2, dim=1)                                  # [B, C, H, W] each

        out = x_normed * (1 + self.gamma_scale * gamma) + self.beta_scale * beta
        return out
class TextConditionedDecoderV2(nn.Module):
    """
    Standard UNet with optional EnhancedStyleTransfer projection after the bottleneck.
    EnhancedStyleTransfer is applied only after the bottleneck, with attention=True.
    """
    def __init__(self, in_ch, out_ch, conditional_ch=0, num_lvs=4, base_ch=16, final_act='noact', use_dropout=True):
        super().__init__()
        self.final_act = final_act
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, 1, 1)
        self.num_lvs = num_lvs
        # --- Style transfer modules ---
        bottleneck_ch = base_ch * (2 ** num_lvs)
        self.enhanced_style_transfer_bottleneck = EnhancedStyleTransferV2(
            style_dim=512, hidden_dim=bottleneck_ch*2, window_size=8, norm_mode="bottleneck_only", use_dropout=use_dropout
        )
        self.enhanced_style_transfer_layers = nn.ModuleList([
            EnhancedStyleTransferV2(style_dim=512, hidden_dim=base_ch * 2 * (2 ** i), window_size=16, norm_mode="bottleneck_only", use_dropout=use_dropout)
            for i in reversed(range(num_lvs))
        ])
        # --- UNet blocks ---
        self.down_convs = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for lv in range(num_lvs):
            ch = base_ch * (2 ** lv)
            self.down_convs.append(ConvBlock2d(ch + conditional_ch, ch * 2, ch * 2))
            self.down_samples.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.up_samples.append(Upsample(ch * 4))
            self.up_convs.append(ConvBlock2d(ch * 4, ch * 2, ch * 2))
        self.bottleneck_conv = ConvBlock2d(bottleneck_ch, bottleneck_ch * 2, bottleneck_ch * 2)
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(base_ch, out_ch, 3, 1, 1)
        )

    def forward(self, in_tensor, condition=None, style_text_feat=None):
        encoded_features = []
        x = self.in_conv(in_tensor)
        for down_conv, down_sample in zip(self.down_convs, self.down_samples):
            if condition is not None:
                feature_dim = x.shape[-1]
                down_conv_out = down_conv(torch.cat([x, condition.repeat(1, 1, feature_dim, feature_dim)], dim=1))
            else:
                down_conv_out = down_conv(x)
            x = down_sample(down_conv_out)
            encoded_features.append(down_conv_out)
        x = self.bottleneck_conv(x)
        x = self.enhanced_style_transfer_bottleneck(x, style_text_feat, use_attention=True, is_bottleneck=True)
        # Upsampling path with per-level style transfer
        for idx, (encoded_feature, up_conv, up_sample, style_layer) in enumerate(
                zip(reversed(encoded_features), reversed(self.up_convs), reversed(self.up_samples), self.enhanced_style_transfer_layers)):
            x = up_sample(x, encoded_feature)
            x = up_conv(x)
            x = style_layer(x, style_text_feat, use_attention=True, is_bottleneck=False)
        x = self.out_conv(x)
        if self.final_act == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.final_act == 'relu':
            x = torch.relu(x)
        elif self.final_act == 'tanh':
            x = torch.tanh(x)
        else:
            x = x
        return x


class EnhancedStyleTransferv3(nn.Module):
    """
    Enhanced style transfer module that better translates text descriptions into visual styles.
    Combines multiple approaches for robust text-to-style transfer.
    """
    def __init__(self, style_dim=512, hidden_dim=256, num_layers=3):
        super().__init__()
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        
        # Multi-scale style encoder
        self.style_encoder = nn.Sequential(
            nn.Linear(style_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Style modulation network
        self.style_modulation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2)
        )
        
       # Multi-head attention for style injection
        self.bottleneck_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=8, 
            batch_first=True
        )
        self.style_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=1, 
            batch_first=True
        )
        
        # Style-aware normalization
        self.style_norm = nn.LayerNorm(hidden_dim)
        


    def forward(self, x, style_text_feat, use_attention=True):
        """
        x: input features [B, C, H, W]
        style_text_feat: text style features [B, style_dim]
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Direct flattening without projection - assume input channels match hidden_dim
        if C != self.hidden_dim:
            # If channels don't match, we need to adapt the input
            # Create a new adaptor for each different input channel count
            adaptor_key = f'input_adaptor_{C}'
            if not hasattr(self, adaptor_key):
                setattr(self, adaptor_key, nn.Conv2d(C, self.hidden_dim, 1).to(device))
            x = getattr(self, adaptor_key)(x)
        
        x_flat = x.view(B, self.hidden_dim, H*W).permute(0, 2, 1)  # [B, H*W, hidden_dim]
        
        # Encode style text features
        style_encoded = self.style_encoder(style_text_feat)  # [B, hidden_dim]
        
        # Create style query from encoded features
        style_query = style_encoded.unsqueeze(1).expand(-1, H*W, -1)  # [B, H*W, hidden_dim]
        
        # Apply style attention only if H*W is small
        if use_attention and H * W < 1024:
            x_attended, _ = self.bottleneck_attention(
                query=style_query,
                key=x_flat,
                value=x_flat
            )
        elif use_attention and H*W > 1024 and H * W < 128*128:  
                x_attended, _ = self.style_attention(
                    query=style_query,
                    key=x_flat,
                    value=x_flat
                )
        else:
            x_attended = x_flat
        # Style modulation
        style_mod = self.style_modulation(style_encoded)  # [B, hidden_dim*2]
        gamma, beta = style_mod.chunk(2, dim=1)  # Each [B, hidden_dim]
        
        # Store for debugging
        self.last_gamma = gamma.detach().cpu() if gamma is not None else None
        self.last_beta = beta.detach().cpu() if beta is not None else None
        
        # Apply normalization first, then modulation
        x_normalized = self.style_norm(x_attended)
        x_modulated = gamma.unsqueeze(1) * 5.0 * x_normalized + beta.unsqueeze(1) * 5.0        
        # Reshape back to spatial format
        x_out = x_modulated.permute(0, 2, 1).view(B, self.hidden_dim, H, W)
        
        return x_out
class TextConditionedDecoderv3(nn.Module):
    """
    Standard UNet with optional EnhancedStyleTransfer projection after the bottleneck.
    EnhancedStyleTransfer is applied only after the bottleneck, with attention=True.
    """
    def __init__(self, in_ch, out_ch, conditional_ch=0, num_lvs=4, base_ch=16, final_act='noact'):
        super().__init__()
        self.final_act = final_act
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, 1, 1)
        self.num_lvs = num_lvs
        # --- Style transfer modules ---
        bottleneck_ch = base_ch * (2 ** num_lvs)
        self.enhanced_style_transfer_bottleneck = EnhancedStyleTransferv3(style_dim=512, hidden_dim=bottleneck_ch*2)
        # For upsampling path, create a style transfer layer for each level, matching upsampled channels
        self.enhanced_style_transfer_layers = nn.ModuleList([
            EnhancedStyleTransferv3(style_dim=512, hidden_dim=base_ch * 2 * (2 ** i))
            for i in reversed(range(num_lvs))
        ])
        # --- UNet blocks ---
        self.down_convs = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for lv in range(num_lvs):
            ch = base_ch * (2 ** lv)
            self.down_convs.append(ConvBlock2d(ch + conditional_ch, ch * 2, ch * 2))
            self.down_samples.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.up_samples.append(Upsample(ch * 4))
            self.up_convs.append(ConvBlock2d(ch * 4, ch * 2, ch * 2))
        self.bottleneck_conv = ConvBlock2d(bottleneck_ch, bottleneck_ch * 2, bottleneck_ch * 2)
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(base_ch, out_ch, 3, 1, 1)
        )

    def forward(self, in_tensor, condition=None, style_text_feat=None):
        encoded_features = []
        x = self.in_conv(in_tensor)
        for down_conv, down_sample in zip(self.down_convs, self.down_samples):
            if condition is not None:
                feature_dim = x.shape[-1]
                down_conv_out = down_conv(torch.cat([x, condition.repeat(1, 1, feature_dim, feature_dim)], dim=1))
            else:
                down_conv_out = down_conv(x)
            x = down_sample(down_conv_out)
            encoded_features.append(down_conv_out)
        x = self.bottleneck_conv(x)
        x = self.enhanced_style_transfer_bottleneck(x, style_text_feat, use_attention=True)
        # Upsampling path with per-level style transfer
        for idx, (encoded_feature, up_conv, up_sample, style_layer) in enumerate(
                zip(reversed(encoded_features), reversed(self.up_convs), reversed(self.up_samples), self.enhanced_style_transfer_layers)):
            x = up_sample(x, encoded_feature)
            x = up_conv(x)
            x = style_layer(x, style_text_feat, use_attention=True)
        x = self.out_conv(x)
        if self.final_act == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.final_act == 'relu':
            x = torch.relu(x)
        elif self.final_act == 'tanh':
            x = torch.tanh(x)
        else:
            x = x
        return x
    
class EnhancedStyleTransfer(nn.Module):
    """
    Enhanced style transfer module that better translates text descriptions into visual styles.
    Combines multiple approaches for robust text-to-style transfer.
    """
    def __init__(self, style_dim=512, hidden_dim=256, num_layers=3):
        super().__init__()
        self.style_dim = style_dim
        self.hidden_dim = hidden_dim
        
        # Multi-scale style encoder
        self.style_encoder = nn.Sequential(
            nn.Linear(style_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Style modulation network
        self.style_modulation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2)
        )
        
        # Multi-head attention for style injection
        self.style_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=8, 
            batch_first=True
        )
        
        # Style-aware normalization
        self.style_norm = nn.LayerNorm(hidden_dim)
        


    def forward(self, x, style_text_feat, use_attention=True):
        """
        x: input features [B, C, H, W]
        style_text_feat: text style features [B, style_dim]
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Direct flattening without projection - assume input channels match hidden_dim
        if C != self.hidden_dim:
            # If channels don't match, we need to adapt the input
            # Create a new adaptor for each different input channel count
            adaptor_key = f'input_adaptor_{C}'
            if not hasattr(self, adaptor_key):
                setattr(self, adaptor_key, nn.Conv2d(C, self.hidden_dim, 1).to(device))
            x = getattr(self, adaptor_key)(x)
        
        x_flat = x.view(B, self.hidden_dim, H*W).permute(0, 2, 1)  # [B, H*W, hidden_dim]
        
        # Encode style text features
        style_encoded = self.style_encoder(style_text_feat)  # [B, hidden_dim]
        
        # Create style query from encoded features
        style_query = style_encoded.unsqueeze(1).expand(-1, H*W, -1)  # [B, H*W, hidden_dim]
        
        # Apply style attention only if H*W is small
        if use_attention and H * W < 1024:
            x_attended, _ = self.style_attention(
                query=style_query,
                key=x_flat,
                value=x_flat
            )
        else:
            if use_attention and H * W > 1024 and not hasattr(self, '_warned_attention'):  # Only warn once
                print(f"[EnhancedStyleTransfer] Skipping attention for shape ({H},{W}) to avoid OOM.")
                self._warned_attention = True
            x_attended = x_flat  # No attention, just pass through
        
        # Style modulation
        style_mod = self.style_modulation(style_encoded)  # [B, hidden_dim*2]
        gamma, beta = style_mod.chunk(2, dim=1)  # Each [B, hidden_dim]
        
        # Store for debugging
        self.last_gamma = gamma.detach().cpu() if gamma is not None else None
        self.last_beta = beta.detach().cpu() if beta is not None else None
        
        # Apply normalization first, then modulation
        x_normalized = self.style_norm(x_attended)
        x_modulated = gamma.unsqueeze(1) * 5.0 * x_normalized + beta.unsqueeze(1) * 5.0        
        # Reshape back to spatial format
        x_out = x_modulated.permute(0, 2, 1).view(B, self.hidden_dim, H, W)
        
        return x_out

class MultiScaleStyleInjector(nn.Module):
    """
    Multi-scale style injection that applies style at different feature scales.
    """
    def __init__(self, in_channels, style_dim=512):
        super().__init__()
        self.in_channels = in_channels
        self.style_dim = style_dim
        
        # Simple linear projections for different scales
        self.style_projections = nn.ModuleList([
            nn.Linear(style_dim, in_channels * 2),
            nn.Linear(style_dim, in_channels * 2),
            nn.Linear(style_dim, in_channels * 2),
            nn.Linear(style_dim, in_channels * 2)
        ])
        
        # Adaptive normalization layers
        self.norm_layers = nn.ModuleList([
            nn.InstanceNorm2d(in_channels),
            nn.InstanceNorm2d(in_channels),
            nn.InstanceNorm2d(in_channels),
            nn.InstanceNorm2d(in_channels)
        ])
        
    def forward(self, features_list, style_text_feat):
        """
        features_list: list of features at different scales
        style_text_feat: text style features [B, style_dim]
        """
        modulated_features = []
        
        for i, (features, style_proj, norm_layer) in enumerate(
            zip(features_list, self.style_projections, self.norm_layers)
        ):
            # Project style features to modulation parameters
            style_params = style_proj(style_text_feat)  # [B, in_channels*2]
            gamma, beta = style_params.chunk(2, dim=1)  # Each [B, in_channels]
            
            # Apply adaptive normalization
            gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
            beta = beta.view(beta.size(0), beta.size(1), 1, 1)
            
            # Normalize and modulate
            features_norm = norm_layer(features)
            features_modulated = gamma * features_norm + beta
            
            modulated_features.append(features_modulated)
        
        return modulated_features
class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            nn.InstanceNorm2d(mid_ch),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, in_tensor):
        return self.conv(in_tensor)
    
class Upsample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        out_ch = in_ch // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, in_tensor, encoded_feature):
        # Match encoded feature size exactly
        B, C, H_skip, W_skip = encoded_feature.shape
        up_sampled_tensor = F.interpolate(in_tensor, size=(H_skip, W_skip), mode='bilinear', align_corners=False)
        up_sampled_tensor = self.conv(up_sampled_tensor)
        return torch.cat([encoded_feature, up_sampled_tensor], dim=1)
class TextConditionedDecoder(nn.Module):
    """
    Standard UNet with optional EnhancedStyleTransfer projection after the bottleneck.
    EnhancedStyleTransfer is applied only after the bottleneck, with attention=True.
    """
    def __init__(self, in_ch, out_ch, conditional_ch=0, num_lvs=4, base_ch=16, final_act='noact'):
        super().__init__()
        self.final_act = final_act
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, 1, 1)
        self.num_lvs = num_lvs
        # --- Style transfer modules ---
        bottleneck_ch = base_ch * (2 ** num_lvs)
        self.enhanced_style_transfer_bottleneck = EnhancedStyleTransfer(style_dim=512, hidden_dim=bottleneck_ch*2)
        # For upsampling path, create a style transfer layer for each level, matching upsampled channels
        self.enhanced_style_transfer_layers = nn.ModuleList([
            EnhancedStyleTransfer(style_dim=512, hidden_dim=base_ch * 2 * (2 ** i))
            for i in reversed(range(num_lvs))
        ])
        # --- UNet blocks ---
        self.down_convs = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for lv in range(num_lvs):
            ch = base_ch * (2 ** lv)
            self.down_convs.append(ConvBlock2d(ch + conditional_ch, ch * 2, ch * 2))
            self.down_samples.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.up_samples.append(Upsample(ch * 4))
            self.up_convs.append(ConvBlock2d(ch * 4, ch * 2, ch * 2))
        self.bottleneck_conv = ConvBlock2d(bottleneck_ch, bottleneck_ch * 2, bottleneck_ch * 2)
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(base_ch, out_ch, 3, 1, 1)
        )

    def forward(self, in_tensor, condition=None, style_text_feat=None):
        encoded_features = []
        x = self.in_conv(in_tensor)
        for down_conv, down_sample in zip(self.down_convs, self.down_samples):
            if condition is not None:
                feature_dim = x.shape[-1]
                down_conv_out = down_conv(torch.cat([x, condition.repeat(1, 1, feature_dim, feature_dim)], dim=1))
            else:
                down_conv_out = down_conv(x)
            x = down_sample(down_conv_out)
            encoded_features.append(down_conv_out)
        x = self.bottleneck_conv(x)
        x = self.enhanced_style_transfer_bottleneck(x, style_text_feat, use_attention=True)
        # Upsampling path with per-level style transfer
        for idx, (encoded_feature, up_conv, up_sample, style_layer) in enumerate(
                zip(reversed(encoded_features), reversed(self.up_convs), reversed(self.up_samples), self.enhanced_style_transfer_layers)):
            x = up_sample(x, encoded_feature)
            x = up_conv(x)
            x = style_layer(x, style_text_feat, use_attention=True)
        x = self.out_conv(x)
        if self.final_act == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.final_act == 'relu':
            x = torch.relu(x)
        elif self.final_act == 'tanh':
            x = torch.tanh(x)
        else:
            x = x
        return x



class MultiScaleDecoder(nn.Module):
    """
    Multi-scale decoder that can handle inputs at different resolutions with skip connections.
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        
        # Multi-scale input processing - each scale gets its own conv layer
        self.scale0_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)  # 224x224
        self.scale1_conv = nn.Conv2d(in_channels, base_channels*2, 3, padding=1)  # 112x112  
        self.scale2_conv = nn.Conv2d(in_channels, base_channels*4, 3, padding=1)  # 56x56
        self.scale3_conv = nn.Conv2d(in_channels, base_channels*8, 3, padding=1)  # 28x28
        
        # Upsampling path with skip connections from all scales
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8, base_channels*4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*4),
            nn.ReLU()
        )
        
        # After concatenating with scale2 features: base_channels*4 + base_channels*4 = base_channels*8
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8, base_channels*2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*2),
            nn.ReLU()
        )
        
        # After concatenating with scale1 features: base_channels*2 + base_channels*2 = base_channels*4
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU()
        )
        
        # After concatenating with scale0 features: base_channels + base_channels = base_channels*2
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
            nn.Sigmoid()  # Ensure output is in [0, 1] range for medical images
        )
    
    def forward(self, multi_scale_features):
        # multi_scale_features: list of [scale0, scale1, scale2, scale3]
        # Each scale has shape [B, in_channels, H, W] where H,W decrease by factor of 2
        scale0, scale1, scale2, scale3 = multi_scale_features
        
        # Process each scale with its dedicated conv layer
        x0 = self.scale0_conv(scale0)  # [B, base_channels, 224, 224]
        x1 = self.scale1_conv(scale1)  # [B, base_channels*2, 112, 112]
        x2 = self.scale2_conv(scale2)  # [B, base_channels*4, 56, 56]
        x3 = self.scale3_conv(scale3)  # [B, base_channels*8, 28, 28]
        
        # Start from coarsest scale and upsample with skip connections
        up3 = self.up3(x3)  # [B, base_channels*4, 56, 56]
        up3 = torch.cat([up3, x2], dim=1)  # Skip connection from scale2: [B, base_channels*8, 56, 56]
        
        up2 = self.up2(up3)  # [B, base_channels*2, 112, 112]
        up2 = torch.cat([up2, x1], dim=1)  # Skip connection from scale1: [B, base_channels*4, 112, 112]
        
        up1 = self.up1(up2)  # [B, base_channels, 224, 224]
        up1 = torch.cat([up1, x0], dim=1)  # Skip connection from scale0: [B, base_channels*2, 224, 224]
        
        output = self.final_conv(up1)  # [B, out_channels, 224, 224]
        return output

class DualCrossAttentionStyleTransfer(nn.Module):
    def __init__(self, style_dim=512, hidden_dim=256, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Style encoder (same as yours)
        self.style_encoder = nn.Sequential(
            nn.Linear(style_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Two cross-attention layers
        self.content_as_q_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.style_as_q_attn   = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Style modulation network
        self.style_modulation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, style_text_feat):
        """
        x: [B, C, H, W] content features
        style_text_feat: [B, style_dim] style embedding from text
        """
        B, C, H, W = x.shape
        if C != self.hidden_dim:
            x = nn.Conv2d(C, self.hidden_dim, 1)(x)
        
        # Flatten spatial dims
        x_flat = x.view(B, self.hidden_dim, H*W).permute(0, 2, 1)  # [B, HW, C]
        
        # Encode style
        style_encoded = self.style_encoder(style_text_feat)        # [B, C]
        style_seq = style_encoded.unsqueeze(1)                     # [B, 1, C]
        
        # --- Attention Pass 1: Content as Q ---
        content_q = x_flat
        style_kv = style_seq.expand(-1, H*W, -1)                   # Broadcast style
        out1, _ = self.content_as_q_attn(query=content_q, key=style_kv, value=style_kv)
        
        # --- Attention Pass 2: Style as Q ---
        style_q = style_seq
        out2, _ = self.style_as_q_attn(query=style_q, key=x_flat, value=x_flat)
        out2 = out2.expand(-1, H*W, -1)                            # Broadcast back
        
        # --- Fuse ---
        fused = self.fusion(torch.cat([out1, out2], dim=-1))       # [B, HW, C]
        
        # --- Modulation ---
        style_mod = self.style_modulation(style_encoded)           # [B, C*2]
        gamma, beta = style_mod.chunk(2, dim=1)                    # [B, C] each
        
        fused = self.norm(fused)
        fused = gamma.unsqueeze(1) * fused + beta.unsqueeze(1)
        
        # Reshape back
        out = fused.permute(0, 2, 1).view(B, self.hidden_dim, H, W)
        return out