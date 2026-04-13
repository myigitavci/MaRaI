import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import List, Optional, Sequence, Tuple, Union
from monai.networks.layers import  Act
from monai.networks.blocks import Convolution


class FusionNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, 8, 3, 1, 1),
            nn.InstanceNorm3d(8),
            nn.LeakyReLU(),
            nn.Conv3d(8, 16, 3, 1, 1),
            nn.InstanceNorm3d(16),
            nn.LeakyReLU())
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_ch + 16, 16, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv3d(16, out_ch, 3, 1, 1),
            nn.ReLU())

    def forward(self, x):
        # return self.conv2(x + self.conv1(x))
        return self.conv2(torch.cat([x, self.conv1(x)], dim=1))

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, conditional_ch=0, num_lvs=4, base_ch=16, final_act='noact'):
        super().__init__()
        self.final_act = final_act
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, 1, 1)
        self.contrast_norm = nn.InstanceNorm2d(in_ch, affine=True)

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
        bottleneck_ch = base_ch * (2 ** num_lvs)
        self.bottleneck_conv = ConvBlock2d(bottleneck_ch, bottleneck_ch * 2, bottleneck_ch * 2)
        self.out_conv = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1),
                                      nn.LeakyReLU(0.1),
                                      nn.Conv2d(base_ch, out_ch, 3, 1, 1))

    def forward(self, in_tensor, condition=None):
        encoded_features = []
       # x = self.contrast_norm(in_tensor)
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
        for encoded_feature, up_conv, up_sample in zip(reversed(encoded_features),
                                                       reversed(self.up_convs),
                                                       reversed(self.up_samples)):
            x = up_sample(x, encoded_feature)
            x = up_conv(x)
        x = self.out_conv(x)
        if self.final_act == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.final_act == "relu":
            x = torch.relu(x)
        elif self.final_act == 'tanh':
            x = torch.tanh(x)
        else:
            x = x
        return x


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
        up_sampled_tensor = F.interpolate(in_tensor, size=None, scale_factor=2, mode='bilinear', align_corners=False)
        up_sampled_tensor = self.conv(up_sampled_tensor)
        return torch.cat([encoded_feature, up_sampled_tensor], dim=1)


class Patchifier(nn.Module):
    def __init__(self, in_ch, out_ch=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 16, 16, 0),  # (*, in_ch, 224, 224) --> (*, 64, 7, 7)
            #nn.Conv2d(in_ch, 64, kernel_size=8, stride=4, padding=2),  # overlapping patches
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, out_ch, 1, 1, 0))

    def forward(self, x):
        return self.conv(x)


class Patchifier3D(nn.Module):
    """
    3D Patchifier that extracts 16x16x16 patches from 3D volumes.
    Input: [B, C, D, H, W] where D, H, W are spatial dimensions
    Output: [B, out_ch, D', H', W'] where each spatial dimension is divided by 16
    """
    def __init__(self, in_ch, out_ch=128):
        super().__init__()
        self.conv = nn.Sequential(
            # Extract 16x16x16 patches with stride 16 (non-overlapping)
            nn.Conv3d(in_ch, 64, kernel_size=16, stride=16, padding=0),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.1),
            nn.Conv3d(64, out_ch, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, D, H, W] - input 3D volume
        Returns:
            features: [B, out_ch, D', H', W'] where D'=D//16, H'=H//16, W'=W//16
        """
        return self.conv(x)


class ThetaEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 17, 9, 4),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1),  # (*, 32, 28, 28)
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1),  # (*, 64, 14, 14)
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.1))  # (* 64, 7, 7)
        self.mean_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, out_ch, 6, 6, 0))
        self.logvar_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, out_ch, 6, 6, 0))

    def forward(self, x):
        M = self.conv(x)
        mu = self.mean_conv(M)
        logvar = self.logvar_conv(M)
        return mu, logvar

# class ThetaEncoder(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, 32, 32, 32, 0),  # (*, in_ch, 224, 244) --> (*, 32, 7, 7)
#             nn.InstanceNorm2d(32),
#             nn.LeakyReLU(0.1),
#             nn.Conv2d(32, 64, 1, 1, 0),
#             # nn.InstanceNorm2d(64),
#             nn.LeakyReLU(0.1))
#         self.mu_conv = nn.Sequential(
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.InstanceNorm2d(64),
#             nn.LeakyReLU(0.1),
#             nn.Conv2d(64, out_ch, 7, 7, 0))
#         self.logvar_conv = nn.Sequential(
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.InstanceNorm2d(64),
#             nn.LeakyReLU(0.1),
#             nn.Conv2d(64, out_ch, 7, 7, 0))
#
#     def forward(self, x, patch_shuffle=False):
#         m = self.conv(x)
#         if patch_shuffle:
#             batch_size = m.shape[0]
#             num_features = m.shape[1]
#             num_patches_per_dim = m.shape[-1]
#             m = m.view(batch_size, num_features, -1)[:, :, torch.randperm(num_patches_per_dim ** 2)]
#             m = m.view(batch_size, num_features, num_patches_per_dim, num_patches_per_dim)
#         mu = self.mu_conv(m)
#         logvar = self.logvar_conv(m)
#         return mu, logvar


class AttentionModule(nn.Module):
    def __init__(self, dim, v_ch=5):
        super().__init__()
        self.dim = dim
        self.v_ch = v_ch
        self.q_fc = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 16),
            nn.LayerNorm(16))
        self.k_fc = nn.Sequential(
            nn.Linear(dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 16),
            nn.LayerNorm(16))

        self.scale = self.dim ** (-0.5)

    def forward(self, q, k, v, modality_dropout=None, temperature=10.0):
        """
        Attention module for optimal anatomy fusion.

        ===INPUTS===
        * q: torch.Tensor (batch_size, feature_dim_q, num_q_patches=1)
            Query variable. In MR-Styler, query is the target theta.
        * k: torch.Tensor (batch_size, feature_dim_k, num_k_patches=1, num_contrasts=4)
            Key variable. In MR-Styler, keys are theta's of source images.
        * v: torch.Tensor (batch_size, self.v_ch=5, num_v_patches=224*224, num_contrasts=4)
            Value variable. In MR-Styler, values are multi-channel logits of source images.
            self.v_ch is the number of beta channels.
        * modality_dropout: torch.Tensor (batch_size, num_contrasts=4)
            Indicates which contrast indexes have been dropped out. 1: if dropped out, 0: if exists.
        """
        batch_size, feature_dim_q, num_q_patches = q.shape
        _, feature_dim_k, _, num_contrasts = k.shape
        num_v_patches = v.shape[2]
        assert (
                feature_dim_k == feature_dim_q or feature_dim_q == self.feature_dim
        ), 'Feature dimensions do not match.'

        # q.shape: (batch_size, num_q_patches=1, 1, feature_dim_q)
        q = q.reshape(batch_size, feature_dim_q, num_q_patches, 1).permute(0, 2, 3, 1)
        # k.shape: (batch_size, num_k_patches=1, num_contrasts=4, feature_dim_k)
        k = k.permute(0, 2, 3, 1)
        # v.shape: (batch_size, num_v_patches=224*224, num_contrasts=4, v_ch=5)
        v = v.permute(0, 2, 3, 1)
        q = self.q_fc(q)
        # k.shape: (batch_size, num_k_patches=1, feature_dim_k, num_contrasts=4)
        k = self.k_fc(k).permute(0, 1, 3, 2)

        # dot_prod.shape: (batch_size, num_q_patches=1, 1, num_contrasts=4)
        dot_prod = (q @ k) * self.scale
        interpolation_factor = int(math.sqrt(num_v_patches // num_q_patches))

        q_spatial_dim = int(math.sqrt(num_q_patches))
        dot_prod = dot_prod.view(batch_size, q_spatial_dim, q_spatial_dim, num_contrasts)

        image_dim = int(math.sqrt(num_v_patches))
        # dot_prod_interp.shape: (batch_size, image_dim, image_dim, num_contrasts)
        dot_prod_interp = dot_prod.repeat(1, interpolation_factor, interpolation_factor, 1)
        if modality_dropout is not None:
            modality_dropout = modality_dropout.view(batch_size, num_contrasts, 1, 1).permute(0, 2, 3, 1)
            dot_prod_interp = dot_prod_interp - (modality_dropout.repeat(1, image_dim, image_dim, 1).detach() * 1e5)

        attention = (dot_prod_interp / temperature).softmax(dim=-1)
        v = attention.view(batch_size, num_v_patches, 1, num_contrasts) @ v
        v = v.view(batch_size, image_dim, image_dim, self.v_ch).permute(0, 3, 1, 2)
        attention = attention.view(batch_size, image_dim, image_dim, num_contrasts).permute(0, 3, 1, 2)
        return v, attention


class FeatureAdapter(nn.Module):
    """
    Adapter that maps a 512-dim CLIP feature (text or image) to a 224x224 spatial map.
    Can be used for both text and image features.
    """
    def __init__(self, in_dim=512, out_size=224):
        super().__init__()
        self.out_size = out_size
        # Project to a small spatial map, then upsample
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 7 * 7),
            nn.ReLU()
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(1, 16, 4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, 4, stride=2, padding=1), # 14x14 -> 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1), # 28x28 -> 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), # 56x56 -> 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),  # 112x112 -> 224x224
        )

    def forward(self, x):
        # x: (batch, 512)
        x = self.mlp(x)  # (batch, 49)
        x = x.view(-1, 1, 7, 7)  # (batch, 1, 7, 7)
        x = self.upsample(x)     # (batch, 1, 224, 224)
        return x


class AdaINBlock(nn.Module):
    """
    Adaptive Instance Normalization block for injecting style/contrast features into a feature map.
    """
    def __init__(self, num_channels, style_dim=512):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_channels)
        self.mlp = nn.Sequential(
            nn.Linear(style_dim, 256),
            nn.LeakyReLU(0.2),   # Slope for negative inputs
            nn.Linear(256, num_channels * 2)
        )

    def forward(self, x, style_vec):
        # x: (B, C, H, W), style_vec: (B, style_dim)
        params = self.mlp(style_vec)  # (B, 2*C)
        gamma, beta = params.chunk(2, dim=1)  # Each (B, C)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        x_norm = self.norm(x)
        return gamma * x_norm + beta
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, context):
        """
        x: [B, N, C] - image tokens
        context: [B, M, C] - text/image guidance tokens
        """
        attn_out, _ = self.cross_attn(query=x, key=context, value=context)
        x = self.norm(x + attn_out)
        return x


class PatchDiscriminator(nn.Sequential):
    """
    Patch-GAN discriminator based on Pix2PixHD:
    High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs
    Ting-Chun Wang1, Ming-Yu Liu1, Jun-Yan Zhu2, Andrew Tao1, Jan Kautz1, Bryan Catanzaro (1)
    (1) NVIDIA Corporation, 2UC Berkeley
    In CVPR 2018.

    Args:
        spatial_dims: number of spatial dimensions (1D, 2D etc.)
        num_channels: number of filters in the first convolutional layer (double of the value is taken from then on)
        in_channels: number of input channels
        out_channels: number of output channels in each discriminator
        num_layers_d: number of Convolution layers (Conv + activation + normalisation + [dropout]) in each
            of the discriminators. In each layer, the number of channels are doubled and the spatial size is
            divided by 2.
        kernel_size: kernel size of the convolution layers
        activation: activation layer type
        norm: normalisation type
        bias: introduction of layer bias
        padding: padding to be applied to the convolutional layers
        dropout: proportion of dropout applied, defaults to 0.
        last_conv_kernel_size: kernel size of the last convolutional layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_channels: int,
        in_channels: int,
        out_channels: int = 1,
        num_layers_d: int = 3,
        kernel_size: int = 4,
        activation: Union[str, tuple] = (Act.LEAKYRELU, {"negative_slope": 0.2}),
        norm: Union[str, tuple] = "BATCH",
        bias: bool = False,
        padding: Union[int, Sequence[int]] = 1,
        dropout: Union[float, tuple] = 0.0,
        last_conv_kernel_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_layers_d = num_layers_d
        self.num_channels = num_channels
        if last_conv_kernel_size is None:
            last_conv_kernel_size = kernel_size

        self.add_module(
            "initial_conv",
            Convolution(
                spatial_dims=spatial_dims,
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=num_channels,
                act=activation,
                bias=True,
                norm=None,
                dropout=dropout,
                padding=padding,
                strides=2,
            ),
        )

        input_channels = num_channels
        output_channels = num_channels * 2

        # Initial Layer
        for l_ in range(self.num_layers_d):
            if l_ == self.num_layers_d - 1:
                stride = 1
            else:
                stride = 2
            layer = Convolution(
                spatial_dims=spatial_dims,
                kernel_size=kernel_size,
                in_channels=input_channels,
                out_channels=output_channels,
                act=activation,
                bias=bias,
                norm=norm,
                dropout=dropout,
                padding=padding,
                strides=stride,
            )
            self.add_module("%d" % l_, layer)
            input_channels = output_channels
            output_channels = output_channels * 2

        # Final layer
        self.add_module(
            "final_conv",
            Convolution(
                spatial_dims=spatial_dims,
                kernel_size=last_conv_kernel_size,
                in_channels=input_channels,
                out_channels=out_channels,
                bias=True,
                conv_only=True,
                padding=int((last_conv_kernel_size - 1) / 2),
                dropout=0.0,
                strides=1,
            ),
        )

        self.apply(self.initialise_weights)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """

        Args:
            x: input tensor
            feature-matching loss (regulariser loss) on the discriminators as well (see Pix2Pix paper).
        Returns:
            list of intermediate features, with the last element being the output.
        """
        out = [x]
        for submodel in self.children():
            intermediate_output = submodel(out[-1])
            out.append(intermediate_output)

        return out[1:]

    def initialise_weights(self, m: nn.Module) -> None:
        """
        Initialise weights of Convolution and BatchNorm layers.

        Args:
            m: instance of torch.nn.module (or of class inheriting torch.nn.module)
        """
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("Conv3d") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("Conv1d") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


class GradientPenaltyLoss(nn.Module):
    """
    Gradient penalty loss for Wasserstein GAN training stability.
    """
    def __init__(self, lambda_gp=10.0):
        super().__init__()
        self.lambda_gp = lambda_gp
    
    def forward(self, discriminator, real_samples, fake_samples):
        """
        Compute gradient penalty loss.
        
        Args:
            discriminator: The discriminator network
            real_samples: Real image samples
            fake_samples: Generated image samples
            
        Returns:
            gradient_penalty: Gradient penalty loss
        """
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, device=real_samples.device)
        
        # Interpolate between real and fake samples
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        # Get discriminator output for interpolated samples
        d_interpolates = discriminator(interpolates)
        
        # Check if gradients can be computed
        if not d_interpolates.requires_grad:
            # If no gradients, return zero penalty
            return torch.tensor(0.0, device=real_samples.device, requires_grad=False)
        
        # Compute gradients
        fake = torch.ones(d_interpolates.size(), device=real_samples.device, requires_grad=False)
        try:
            gradients = torch.autograd.grad(
                outputs=d_interpolates,
                inputs=interpolates,
                grad_outputs=fake,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            
            # Compute gradient penalty
            gradients = gradients.view(batch_size, -1)
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            
            return gradient_penalty * self.lambda_gp
        except RuntimeError:
            # If gradient computation fails, return zero penalty
            return torch.tensor(0.0, device=real_samples.device, requires_grad=False)