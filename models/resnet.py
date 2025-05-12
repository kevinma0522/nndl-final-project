"""
Implementation of ResNet50 for climate forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ClimateResNet(nn.Module):
    def __init__(self, input_channels, output_channels, pretrained=True):
        """
        Initialize the ClimateResNet model.
        
        Args:
            input_channels (int): Number of input channels
            output_channels (int): Number of output channels
            pretrained (bool): Whether to use pretrained weights
        """
        super(ClimateResNet, self).__init__()
        
        # Load pretrained ResNet50 and modify for climate data
        self.resnet = resnet50(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, output_channels)
        
        # Add attention mechanism
        self.attention = MultiHeadSelfAttention(num_features)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_channels)
        """
        # Extract features through ResNet layers
        features = self.resnet.conv1(x)
        features = self.resnet.bn1(features)
        features = self.resnet.relu(features)
        features = self.resnet.maxpool(features)
        
        features = self.resnet.layer1(features)
        features = self.resnet.layer2(features)
        features = self.resnet.layer3(features)
        features = self.resnet.layer4(features)
        
        # Apply attention and final classification
        features = self.attention(features)
        features = self.resnet.avgpool(features)
        features = torch.flatten(features, 1)
        output = self.resnet.fc(features)
        
        return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        """
        Multi-head self-attention module.
        
        Args:
            dim (int): Input dimension
            num_heads (int): Number of attention heads
            qkv_bias (bool): Whether to use bias in qkv projection
            attn_drop (float): Dropout rate for attention weights
            proj_drop (float): Dropout rate for output projection
        """
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Forward pass through the attention module.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor with attention applied
        """
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        qkv = self.qkv(x).reshape(B, H*W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x 