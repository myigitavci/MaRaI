import torch
import torch.nn as nn
import torch.nn.functional as F
from dist_clip.modules.model_new import UNet

class SimpleBetaEncoder(nn.Module):
    """
    Simple Beta Encoder that learns underlying anatomy independent of contrast.
    
    Key features:
    1. Contrast normalization to remove contrast variations
    2. Simple convolutional architecture
    3. Focus on anatomical structure learning
    4. Lightweight and fast
    """
    
    def __init__(self, in_channels=1, beta_dim=16, base_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.beta_dim = beta_dim
        self.base_channels = base_channels
        
        # Contrast normalization layer
        self.contrast_norm = nn.InstanceNorm2d(in_channels, affine=True)
        
        # Simple encoder: extract anatomical features
        self.encoder = nn.Sequential(
            # Initial conv with larger kernel to capture broader patterns
            nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Downsample and extract features
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Process features
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Downsample again
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # Final processing
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
        )
        
        # Upsample back to original resolution
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        
        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, beta_dim, kernel_size=1),
        )
        

    
    def forward(self, x):
        """
        Forward pass: learn anatomy independent of contrast
        
        Args:
            x: Input image [B, 1, H, W]
            
        Returns:
            beta_logits: Raw logits [B, beta_dim, H, W]
            beta_probs: Probability distribution [B, beta_dim, H, W]
        """
        # Step 1: Contrast normalization to remove contrast variations
        x_normalized = self.contrast_norm(x)
        
        # Step 2: Extract anatomical features
        features = self.encoder(x_normalized)
        
        # Step 3: Upsample to original resolution
        upsampled = self.upsampler(features)
        
        # Step 4: Generate beta encoding
        beta_logits = self.output_layer(upsampled)
        
        # Step 5: Convert to probabilities (optional, for compatibility)
        beta_probs = F.softmax(beta_logits, dim=1)
        
        return beta_logits, beta_probs
    
    def get_anatomical_features(self, x):
        """
        Extract anatomical features without final classification
        
        Args:
            x: Input image [B, 1, H, W]
            
        Returns:
            features: Anatomical features [B, base_channels*4, H/4, W/4]
        """
        x_normalized = self.contrast_norm(x)
        features = self.encoder(x_normalized)
        return features


class MinimalBetaEncoder(nn.Module):
    """
    Minimal version for very fast inference
    """
    
    def __init__(self, in_channels=1, beta_dim=16, base_channels=16):
        super().__init__()
        
        self.contrast_norm = nn.InstanceNorm2d(in_channels, affine=True)
        
        self.encoder = nn.Sequential(
            # Simple encoder
            nn.Conv2d(in_channels, base_channels, 7, padding=3),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Upsample
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Output
            nn.Conv2d(base_channels, beta_dim, 1),
        )
    
    def forward(self, x):
        x_normalized = self.contrast_norm(x)
        beta_logits = self.encoder(x_normalized)
        beta_probs = F.softmax(beta_logits, dim=1)
        return beta_logits, beta_probs


def create_simple_beta_encoder(beta_dim, encoder_type='simple', **kwargs):
    """
    Factory function to create simple beta encoders
    
    Args:
        beta_dim: Number of beta classes
        encoder_type: 'simple', 'minimal', 'old', or 'brain_tissue'
        **kwargs: Additional arguments
    """
    if encoder_type == 'simple':
        return SimpleBetaEncoder(beta_dim=beta_dim, **kwargs)
    elif encoder_type == 'minimal':
        return MinimalBetaEncoder(beta_dim=beta_dim, **kwargs)
    elif encoder_type == 'old':
        return UNet(in_ch=1, out_ch=beta_dim,num_lvs=4, base_ch=kwargs['base_channels'], final_act='sigmoid')
    elif encoder_type == 'brain_tissue':
        from dist_clip.modules.brain_tissue_beta_encoder import create_brain_tissue_beta_encoder
        return create_brain_tissue_beta_encoder(beta_dim=beta_dim, **kwargs)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")


# Utility functions
def compute_anatomy_consistency_loss(beta1, beta2, temperature=0.1):
    """
    Compute consistency loss between anatomical features
    """
    # Flatten and normalize
    beta1_flat = F.normalize(beta1.view(beta1.size(0), -1), dim=1)
    beta2_flat = F.normalize(beta2.view(beta2.size(0), -1), dim=1)
    
    # Compute cosine similarity
    similarity = torch.cosine_similarity(beta1_flat, beta2_flat, dim=1)
    
    # Consistency loss (maximize similarity)
    loss = 1 - similarity.mean()
    
    return loss


def visualize_beta_encoding(beta_probs, save_path=None):
    """
    Visualize beta encoding as a colored map
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert to numpy
    beta_np = beta_probs.detach().cpu().numpy()
    
    # Take argmax to get class labels
    beta_labels = np.argmax(beta_np, axis=1)
    
    # Create visualization
    fig, axes = plt.subplots(1, beta_np.shape[0], figsize=(4*beta_np.shape[0], 4))
    if beta_np.shape[0] == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        im = ax.imshow(beta_labels[i], cmap='tab20', vmin=0, vmax=beta_probs.size(1)-1)
        ax.set_title(f'Sample {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 