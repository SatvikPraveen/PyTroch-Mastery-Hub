# src/computer_vision/models.py
"""
CV-specific models for PyTorch Mastery Hub
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from ..neural_networks.layers import ConvLayer, ResidualBlock, AttentionLayer


class SimpleCNN(nn.Module):
    """
    Simple CNN for image classification.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        filters: List[int] = [32, 64, 128, 256],
        dropout: float = 0.5
    ):
        super(SimpleCNN, self).__init__()
        
        # Feature extractor
        layers = []
        in_channels = input_channels
        
        for out_channels in filters:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            ])
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(filters[-1], num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNetCV(nn.Module):
    """
    ResNet for computer vision tasks.
    """
    
    def __init__(
        self,
        block_layers: List[int] = [2, 2, 2, 2],
        num_classes: int = 1000,
        input_channels: int = 3,
        width_multiplier: float = 1.0
    ):
        super(ResNetCV, self).__init__()
        
        base_width = int(64 * width_multiplier)
        self.in_channels = base_width
        
        # Initial layers
        self.conv1 = nn.Conv2d(input_channels, base_width, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # Residual layers
        self.layer1 = self._make_layer(base_width, block_layers[0])
        self.layer2 = self._make_layer(base_width * 2, block_layers[1], stride=2)
        self.layer3 = self._make_layer(base_width * 4, block_layers[2], stride=2)
        self.layer4 = self._make_layer(base_width * 8, block_layers[3], stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, channels: int, num_blocks: int, stride: int = 1):
        downsample = None
        if stride != 1 or self.in_channels != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        
        layers = []
        layers.append(ResidualBlock(self.in_channels, channels, stride, downsample))
        self.in_channels = channels
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(channels, channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class UNet(nn.Module):
    """
    U-Net for semantic segmentation.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512]
    ):
        super(UNet, self).__init__()
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        # Encoder (downsampling)
        in_ch = in_channels
        for feature in features:
            self.encoder.append(self._double_conv(in_ch, feature))
            in_ch = feature
        
        # Bottleneck
        self.bottleneck = self._double_conv(features[-1], features[-1] * 2)
        
        # Decoder (upsampling)
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, 2, 2)
            )
            self.decoder.append(self._double_conv(feature * 2, feature))
        
        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
    
    def _double_conv(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        
        # Encoder
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        # Decoder
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)  # Upsample
            skip_connection = skip_connections[idx // 2]
            
            # Handle size mismatch
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)  # Double conv
        
        return self.final_conv(x)


class FeatureExtractor(nn.Module):
    """
    Feature extractor from pre-trained models.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        feature_layer: Optional[str] = None
    ):
        super(FeatureExtractor, self).__init__()
        
        import torchvision.models as models
        
        # Load backbone
        if backbone == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == 'vgg16':
            self.model = models.vgg16(pretrained=pretrained)
            self.feature_dim = 4096
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove classifier
        if hasattr(self.model, 'fc'):
            self.model.fc = nn.Identity()
        elif hasattr(self.model, 'classifier'):
            self.model.classifier = nn.Identity()
        
        # Hook for intermediate features
        self.features = {}
        self.feature_layer = feature_layer
        
        if feature_layer:
            self._register_hooks()
    
    def _register_hooks(self):
        def hook(name):
            def fn(module, input, output):
                self.features[name] = output
            return fn
        
        for name, module in self.model.named_modules():
            if name == self.feature_layer:
                module.register_forward_hook(hook(name))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_layer:
            self.features.clear()
            _ = self.model(x)
            return self.features[self.feature_layer]
        else:
            return self.model(x)


class ObjectDetector(nn.Module):
    """
    Simple object detector with FCOS-style architecture.
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone_channels: List[int] = [256, 512, 1024, 2048],
        fpn_channels: int = 256
    ):
        super(ObjectDetector, self).__init__()
        
        self.num_classes = num_classes
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(backbone_channels, fpn_channels)
        
        # Detection heads
        self.cls_head = self._make_head(fpn_channels, num_classes)
        self.box_head = self._make_head(fpn_channels, 4)  # x1, y1, x2, y2
        self.centerness_head = self._make_head(fpn_channels, 1)
    
    def _make_head(self, in_channels: int, out_channels: int):
        layers = []
        for _ in range(4):
            layers.extend([
                nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract multi-scale features
        features = self.fpn(x)
        
        cls_logits = []
        box_regression = []
        centerness = []
        
        for feature in features:
            cls_logits.append(self.cls_head(feature))
            box_regression.append(self.box_head(feature))
            centerness.append(self.centerness_head(feature))
        
        return {
            'cls_logits': cls_logits,
            'box_regression': box_regression,
            'centerness': centerness
        }


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network for multi-scale feature extraction.
    """
    
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super(FeaturePyramidNetwork, self).__init__()
        
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, 1)
            output_conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # Top-down pathway
        results = []
        prev_features = self.lateral_convs[-1](features[-1])
        results.append(self.output_convs[-1](prev_features))
        
        for i in range(len(features) - 2, -1, -1):
            lateral = self.lateral_convs[i](features[i])
            
            # Upsample and add
            prev_features = F.interpolate(
                prev_features, size=lateral.shape[-2:], mode='nearest'
            )
            prev_features = lateral + prev_features
            results.insert(0, self.output_convs[i](prev_features))
        
        return results


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) implementation.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.1,
        in_channels: int = 3
    ):
        super(VisionTransformer, self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, d_model) * 0.02
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, d_model, H//p, W//p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Use class token
        
        return self.head(cls_token_final)


class TransformerBlock(nn.Module):
    """
    Transformer block for Vision Transformer.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1
    ):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class EfficientNet(nn.Module):
    """
    Simplified EfficientNet implementation.
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        depth_mult: float = 1.0
    ):
        super(EfficientNet, self).__init__()
        
        # Stem
        stem_channels = int(32 * width_mult)
        self.stem = ConvLayer(3, stem_channels, 3, 2, 1, norm='batch', activation='swish')
        
        # Blocks configuration: (expand_ratio, channels, num_layers, stride)
        blocks_config = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 40, 2, 2),
            (6, 80, 3, 2),
            (6, 112, 3, 1),
            (6, 192, 4, 2),
            (6, 320, 1, 1),
        ]
        
        # Build blocks
        self.blocks = nn.ModuleList()
        in_channels = stem_channels
        
        for expand_ratio, channels, num_layers, stride in blocks_config:
            out_channels = int(channels * width_mult)
            layers = int(num_layers * depth_mult)
            
            for i in range(layers):
                block_stride = stride if i == 0 else 1
                self.blocks.append(
                    MBConvBlock(in_channels, out_channels, expand_ratio, block_stride)
                )
                in_channels = out_channels
        
        # Head
        head_channels = int(1280 * width_mult)
        self.head_conv = ConvLayer(in_channels, head_channels, 1, 1, 0, norm='batch', activation='swish')
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(head_channels, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head_conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Conv Block.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        stride: int,
        se_ratio: float = 0.25
    ):
        super(MBConvBlock, self).__init__()
        
        self.use_residual = stride == 1 and in_channels == out_channels
        expanded_channels = in_channels * expand_ratio
        
        # Expansion
        if expand_ratio != 1:
            self.expand = ConvLayer(in_channels, expanded_channels, 1, 1, 0, norm='batch', activation='swish')
        else:
            self.expand = nn.Identity()
        
        # Depthwise
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, 3, stride, 1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # Squeeze-and-excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expanded_channels, se_channels, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_channels, expanded_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.se = None
        
        # Projection
        self.project = ConvLayer(expanded_channels, out_channels, 1, 1, 0, norm='batch', activation=None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Expansion
        x = self.expand(x)
        
        # Depthwise
        x = self.depthwise(x)
        
        # Squeeze-and-excitation
        if self.se is not None:
            x = x * self.se(x)
        
        # Projection
        x = self.project(x)
        
        # Residual connection
        if self.use_residual:
            x = x + identity
        
        return x