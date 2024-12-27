import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict

class Patch_Embedding(nn.Module):
    def __init__(self, patch_size=16, pic_size=224, in_channels=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.num_patches = (pic_size // patch_size) ** 2
    
    def forward(self, x:torch.Tensor):
        # x:[B, 3, 224, 224] [B, C, H, W]
        B, C, H, W = x.shape
        x = self.conv(x)
        # x:[B, 768, 14, 14] [B, C', H', W']
        x = x.flatten(2)
        # x:[B, 768, 196] [B, C', H'W']
        x = x.transpose(1,2)
        # x:[B, 196, 768] [B, H'W', C']
        x = self.norm(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=None, output_dim=768, drop=0.1):
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHead_Attention(nn.Module):
    def __init__(self, dim=768, num_heads=8, attn_drop=0.0, proj_drop=0.0, qk_scale=None, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # input:[batch_size, num_patches + 1(197), embed_dim(786)]
        B, N, C = x.shape
        # qkv(x).reshape():[batch_size, num_pathches+1, 3, num_heads, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads)
        # qkv.permute():[3, batch_size, num_heads, num_pathces + 1, embed_dim_per_head]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # attn:[batch_size, num_heads, num_patches + 1, num_patches + 1]
        x = (attn @ v).transpose(1, 2)
        # x:[batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        x = x.reshape(B, N, C)
        # x:[batch_size, num_patches + 1(197), embed_dim(786)]
        x = self.proj_drop(self.proj(x))
        return x


class Encoder_Block(nn.Module):
    def __init__(self, dim=768, num_heads=8, mlp_ratio=4.0,attn_drop=0.0, proj_drop=0.0, qk_scale=None, qkv_bias=False, transformer_drop=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.attn = MultiHead_Attention(dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop, qk_scale=qk_scale, qkv_bias=qkv_bias)
        self.dropout = nn.Dropout(transformer_drop)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(input_dim=dim, hidden_dim=mlp_hidden_dim, output_dim=dim)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln(x)))
        x = x + self.dropout(self.mlp(self.ln(x)))
        return x
    

class Transformer_Encoder(nn.Module):
    def __init__(self, num_layers=12, dim=768, mlp_ratio=4.0, num_heads=8, attn_drop=0.0, proj_drop=0.0, qk_scale=None, qkv_bias=False, transformer_drop=0.0):
        super().__init__()
        # 创建多个 Encoder_Block 的堆叠
        self.blocks = nn.ModuleList([
            Encoder_Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop, proj_drop=proj_drop, qk_scale=qk_scale, qkv_bias=qkv_bias, transformer_drop=transformer_drop)
            for _ in range(num_layers)  # num_layers=12
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    

class VisionTransformer(nn.Module):
    def __init__(self, 
                 patch_size=16, 
                 pic_size=224, 
                 in_channels=3, 
                 embed_dim=768, 
                 patch_embed_norm_layer=None, # norm_layer for patch_embedding
                 norm_layer = None, # norm_layer for vit
                 num_layers=12, 
                 num_heads=8,
                 pos_drop=0.0, 
                 attn_drop=0.0, 
                 proj_drop=0.0, 
                 qk_scale=None, 
                 qkv_bias=False, 
                 transformer_drop=0.0,
                 representation_size=None,
                 num_classes=1000):
        super().__init__()
        self.patch_embedding = Patch_Embedding(patch_size=patch_size, pic_size=pic_size, in_channels=in_channels, embed_dim=embed_dim, norm_layer=patch_embed_norm_layer)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6) 
        # class_token and Position Embedding
        self.num_features = embed_dim
        self.num_tokens = 1
        self.num_patches = self.patch_embedding.num_patches
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, embed_dim))
        self.pos_dropout = nn.Dropout(pos_drop)
        self.transformer = Transformer_Encoder(num_layers=num_layers, dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=proj_drop, qk_scale=qk_scale, qkv_bias=qkv_bias, transformer_drop=transformer_drop)
        self.norm = norm_layer(embed_dim)
        
        # representation layer
        if representation_size != None:
            self.num_features = representation_size
            self.representation_layer = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim,representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.representation_layer = nn.Identity()
        self.classification_head = nn.Linear(self.num_features, num_classes)
        # weight init
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.apply(init_vit_weights)

    def forward(self, x):
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        # class_toekn:[B, 1, 768]
        x = self.patch_embedding(x)
        x = torch.concat((class_token, x), dim=1)
        x = self.pos_embedding + x
        x = self.pos_dropout(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = self.representation_layer(x)[:, 0]
        x = self.classification_head(x)
        return x


def init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
    """
    model = VisionTransformer(pic_size=224,
                              patch_size=16,
                              embed_dim=768,
                              num_layers=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)

    return model


def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
    """
    model = VisionTransformer(pic_size=224,
                              patch_size=32,
                              embed_dim=768,
                              num_layers=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)

    return model


def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
    """
    model = VisionTransformer(pic_size=224,
                              patch_size=16,
                              embed_dim=1024,
                              num_layers=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)

    return model


def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    weights ported from official Google JAX impl:
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
    """
    model = VisionTransformer(pic_size=224,
                              patch_size=32,
                              embed_dim=1024,
                              num_layers=24,
                              num_heads=16,
                              representation_size=1024 if has_logits else None,
                              num_classes=num_classes)

    return model


def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
    """
    ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: converted weights not currently available, too large for github release hosting.
    """
    model = VisionTransformer(pic_size=224,
                              patch_size=14,
                              embed_dim=1280,
                              num_layers=32,
                              num_heads=16,
                              representation_size=1280 if has_logits else None,
                              num_classes=num_classes)

    return model
