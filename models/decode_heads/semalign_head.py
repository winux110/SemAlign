from ast import Gt
import numpy as np
from mmcv.cnn import ConvModule
from mmseg.ops import Upsample, resize

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def patch_divide_vectorized(x, step, ps):
    """Divide a feature map into overlapping patches with unfold."""
    B, C, H, W = x.shape
    
    pad_h = (ps - H % step) % step if H % step != 0 else 0
    pad_w = (ps - W % step) % step if W % step != 0 else 0
    
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    
    H_padded, W_padded = x.shape[2], x.shape[3]
    
    patches = x.unfold(2, ps, step).unfold(3, ps, step)
    
    nh, nw = patches.shape[2], patches.shape[3]
    
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.reshape(B, nh * nw, C, ps, ps)
    
    return patches, nh, nw


def patch_reverse_vectorized(crop_x, x, step, ps, nh, nw):
    """Merge overlapping patches back into a feature map by averaging overlaps."""
    B, N, C, ph, pw = crop_x.shape
    H, W = x.shape[2], x.shape[3]
    
    H_out = (nh - 1) * step + ps
    W_out = (nw - 1) * step + ps
    
    crop_x = crop_x.reshape(B, nh, nw, C, ph, pw)
    crop_x = crop_x.permute(0, 3, 4, 5, 1, 2).contiguous()
    crop_x = crop_x.reshape(B, C * ph * pw, nh * nw)
    
    output = F.fold(crop_x, output_size=(H_out, W_out), kernel_size=ps, stride=step)
    
    ones = torch.ones(1, 1, H_out, W_out, device=x.device, dtype=x.dtype)
    divisor = F.unfold(ones, kernel_size=ps, stride=step)
    divisor = F.fold(divisor, output_size=(H_out, W_out), kernel_size=ps, stride=step)
    divisor = divisor.clamp(min=1.0)
    
    output = output / divisor
    
    output = output[:, :, :H, :W]
    
    return output


def patch_reverse_legacy(crop_x, x, step, ps):
    """Reference loop implementation of patch_reverse."""
    b, c, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for i in range(0, h + step - ps, step):
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            output[:, :, top:down, left:right] += crop_x[:, index]
            index += 1
    for i in range(step, h + step - ps, step):
        top = i
        down = i + ps - step
        if top + ps > h:
            top = h - ps
        output[:, :, top:down, :] /= 2
    for j in range(step, w + step - ps, step):
        left = j
        right = j + ps - step
        if left + ps > w:
            left = w - ps
        output[:, :, :, left:right] /= 2
    return output


def patch_divide_legacy(x, step, ps):
    """Reference loop implementation of patch_divide."""
    b, c, h, w = x.size()
    patches = []
    nh, nw = 0, 0
    for i in range(0, h + step - ps, step):
        nh += 1
        nw = 0
        top = i
        down = i + ps
        if down > h:
            top = h - ps
            down = h
        for j in range(0, w + step - ps, step):
            nw += 1
            left = j
            right = j + ps
            if right > w:
                left = w - ps
                right = w
            patches.append(x[:, :, top:down, left:right])
    patches = torch.stack(patches, 1)
    return patches, nh, nw


patch_divide = patch_divide_vectorized
patch_reverse = patch_reverse_vectorized




class ContextAwareTextAdaptation(nn.Module):
    """
    Context-Aware Text Adaptation (CATA).

    Args:
        vis_dim (int): Visual feature dimension
        text_dim (int): Text embedding dimension
        reduction (int): Reduction ratio for SE-style bottleneck (default: 4)
    """
    
    def __init__(self, vis_dim, text_dim, reduction=4):
        super().__init__()
        self.vis_dim = vis_dim
        self.text_dim = text_dim
        
        hidden_dim = max(text_dim // reduction, 64)
        self.vis_to_text = nn.Sequential(
            nn.Linear(vis_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, text_dim)
        )
        
        self.gate = nn.Sequential(
            nn.Linear(text_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, text_dim),
            nn.Sigmoid()
        )
        
        self.scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, visual_feat, text_emb):
        """
        Args:
            visual_feat (Tensor): Global pooled visual features [B, vis_dim]
            text_emb (Tensor): Text embeddings [N_cls, text_dim]
        
        Returns:
            enhanced_text (Tensor): Context-adapted text embeddings [B, N_cls, text_dim]
        """
        B = visual_feat.size(0)
        N_cls = text_emb.size(0)
        
        vis_in_text_space = self.vis_to_text(visual_feat)
        
        text_expanded = text_emb.unsqueeze(0).expand(B, -1, -1)
        
        vis_expanded = vis_in_text_space.unsqueeze(1).expand(-1, N_cls, -1)
        
        combined = torch.cat([text_expanded, vis_expanded], dim=-1)
        gate = self.gate(combined)
        
        enhanced_text = text_expanded + self.scale.to(dtype=text_expanded.dtype) * gate * vis_expanded
        
        return enhanced_text


class ContextAwareTextAdaptationLite(nn.Module):
    """
    Lightweight CATA with a gate shared across classes.

    Args:
        vis_dim (int): Visual feature dimension
        text_dim (int): Text embedding dimension
    """
    
    def __init__(self, vis_dim, text_dim):
        super().__init__()
        self.vis_dim = vis_dim
        self.text_dim = text_dim
        
        self.vis_to_text = nn.Linear(vis_dim, text_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(vis_dim, text_dim),
            nn.Sigmoid()
        )
        
        self.scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, visual_feat, text_emb):
        """
        Args:
            visual_feat (Tensor): Global pooled visual features [B, vis_dim]
            text_emb (Tensor): Text embeddings [N_cls, text_dim]
        
        Returns:
            enhanced_text (Tensor): Context-adapted text embeddings [B, N_cls, text_dim]
        """
        B = visual_feat.size(0)
        N_cls = text_emb.size(0)
        
        vis_in_text_space = self.vis_to_text(visual_feat)
        
        gate = self.gate(visual_feat)
        
        text_expanded = text_emb.unsqueeze(0).expand(B, -1, -1)
        vis_expanded = vis_in_text_space.unsqueeze(1)
        gate_expanded = gate.unsqueeze(1)
        
        enhanced_text = text_expanded + self.scale.to(dtype=text_expanded.dtype) * gate_expanded * vis_expanded
        
        return enhanced_text



class PreNorm(nn.Module):
    """Normalization layer.
    Args:
        dim (int): Base channels.
        fn (Module): Module after normalization.
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).reshape(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x

class LanguageGuidedAnisotropicAttention(nn.Module):
    """
    Language-guided Anisotropic Attention with text-conditioned Q/K gates.

    Args:
        dim (int): Base channels.
        heads (int): Head numbers.
        qk_dim (int): Channels of query and key.
        text_dim (int): Dimension of text embeddings. If None, no text guidance.
        use_efm (bool): Whether to use Enhanced FiLM (Tanh+Shift). Default: True.
    """

    def __init__(self, dim, heads, qk_dim, text_dim=None, use_efm=False):
        super().__init__()

        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5
        self.use_efm = use_efm

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
        self.use_text_mod = text_dim is not None
        if self.use_text_mod:
            self.text_proj_q = nn.Linear(text_dim, qk_dim)
            self.text_proj_k = nn.Linear(text_dim, qk_dim)

    def forward(self, x, text_emb=None):
        """
        Args:
            x: Visual features [B, N, C]
            text_emb: Pooled text embedding [B, text_dim] or None
        Returns:
            out: Attended features [B, N, C]
        """
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        
        if self.use_text_mod and text_emb is not None:
            proj_q = self.text_proj_q(text_emb)
            proj_k = self.text_proj_k(text_emb)
            
            if self.use_efm:
                gate_q = (1.0 + 0.5 * torch.tanh(proj_q)).unsqueeze(1)
                gate_k = (1.0 + 0.5 * torch.tanh(proj_k)).unsqueeze(1)
            else:
                gate_q = torch.sigmoid(proj_q).unsqueeze(1)
                gate_k = torch.sigmoid(proj_k).unsqueeze(1)
            
            q = q * gate_q
            k = k * gate_k
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)

class CostRefiner(nn.Module):
    """Residual cost refinement with a learnable gate."""
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1)
        )
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, cost):
        return cost + self.gate.to(dtype=cost.dtype) * self.conv(cost)


class LAA(nn.Module):
    """
    Language-guided Anisotropic Attention (LAA).

    Args:
        dim: Feature dimension
        qk_dim: Query/Key dimension
        mlp_dim: FFN hidden dimension
        heads: Number of attention heads
        ps: Default patch size
        text_dim: Text embedding dimension (enables language guidance if provided)
        use_cata: Whether to use Context-Aware Text Adaptation (default: True)
        use_efm: Whether to use Enhanced FiLM modulation (default: True)
        cata_lite: Whether to use lightweight CATA variant (default: False)
    """
    def __init__(self, dim, qk_dim, mlp_dim, heads=1, ps=4, text_dim=None,
                 use_cata=True, use_efm=False, cata_lite=False):
        super().__init__()
        self.ps_table = {0: 8, 1: 6, 2: 4, 3: 4, 4: 2, 5: 2}
        self.default_ps = ps
        
        self.text_dim = text_dim
        self.use_cata = use_cata and (text_dim is not None)
        self.use_efm = use_efm
        
        if self.use_cata:
            if cata_lite:
                self.cata = ContextAwareTextAdaptationLite(vis_dim=dim, text_dim=text_dim)
            else:
                self.cata = ContextAwareTextAdaptation(vis_dim=dim, text_dim=text_dim)

        if text_dim is not None:
            self.text_pool_attn = nn.MultiheadAttention(embed_dim=text_dim, num_heads=4, batch_first=True)
            self.vis_query_proj = nn.Linear(dim, text_dim)
        
        self.layer = nn.ModuleList([
            PreNorm(dim, LanguageGuidedAnisotropicAttention(dim, heads, qk_dim, text_dim=text_dim, use_efm=use_efm)),
            PreNorm(dim, ConvFFN(dim, mlp_dim))
        ])
        
        self.res_scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, stage_idx=None, text_token=None):
        """
        Args:
            x: Visual feature map [B, C, H, W]
            stage_idx: Index to select patch size from ps_table
            text_token: Text embeddings [N_cls, dim]. Will be pooled to [1, dim]
        Returns:
            Enhanced feature map [B, C, H, W]
        """
        org = x
        B, C, H, W = x.shape
        
        adapted_text_token = text_token
        if self.use_cata and text_token is not None:
            visual_context = x.mean(dim=[2, 3])
            
            text_token_aligned = text_token.to(dtype=visual_context.dtype)
            
            adapted_text_token = self.cata(visual_context, text_token_aligned)
        
        ps = self.ps_table.get(stage_idx, self.default_ps)
        step = ps - 2
        
        crop_x, nh, nw = patch_divide(x, step, ps)
        b, n, c, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c h w -> (b n) (h w) c')

        attn, ff = self.layer
        
        text_emb = None
        if text_token is not None and self.text_dim is not None:
            if self.use_cata and adapted_text_token is not None and adapted_text_token.dim() == 3:
                vis_query = self.vis_query_proj(x.mean(dim=[2, 3])).unsqueeze(1)
                
                text_emb_pooled, _ = self.text_pool_attn(vis_query, adapted_text_token, adapted_text_token)
                text_emb = text_emb_pooled.squeeze(1)

                text_emb = text_emb.unsqueeze(1).expand(-1, n, -1).reshape(b * n, -1)
            else:
                text_token_aligned = text_token.to(dtype=crop_x.dtype)
                text_emb = text_token_aligned.mean(dim=0, keepdim=True)
                text_emb = text_emb.expand(b * n, -1)
        
        crop_x = attn(crop_x, text_emb=text_emb) + crop_x
        crop_x = rearrange(crop_x, '(b n) (h w) c -> b n c h w', n=n, w=pw)

        x = patch_reverse(crop_x, org, step, ps, nh, nw)
        
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = ff(x, x_size=(h, w)) + x
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)

        return org + self.res_scale.to(dtype=org.dtype) * x

from mmcv.runner import auto_fp16, force_fp32
import matplotlib.pyplot as plt

from timm.models.layers import trunc_normal_
import matplotlib.pyplot as plt
from mmseg.models.losses import accuracy

from models.decode_heads.utils import positional_encoding

def gaussian(x, mu, sigma):
    return torch.exp(-0.5 * (x / sigma)**2)

def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class TPN_Decoder(TransformerDecoder):
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        output = tgt
        attns = []
        outputs = []
        for mod in self.layers:
            output, attn = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            attns.append(attn)
            outputs.append(output)
        if self.norm is not None:
            output = self.norm(output)

        return outputs, attns

class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        tgt2, attn2 = self.multihead_attn(
            tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):
        B, Nq, C = xq.size() 
        Nk = xk.size()[1]
        Nv = xv.size()[1]
        q = self.q(xq).reshape(B, Nq, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(xk).reshape(B, Nk, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.transpose(0, 1), attn_save.sum(dim=1) / self.num_heads

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@HEADS.register_module()
class SemAlignHead(BaseDecodeHead):
    """Segmentation head with LAA, CATA, and cost propagation."""
    def __init__(
            self,
            img_size,
            in_channels,
            seen_idx,
            all_idx,
            embed_dims=768,
            num_layers=3,
            num_heads=8,
            use_stages=5,
            use_laa=True,
            laa_stages=[2,3],
            laa_heads=4,
            laa_patch_size=4,
            text_dim=768,
            use_cata=True,
            use_efm=False,
            cata_lite=False,
            use_ccp=True,
            **kwargs,
    ):
        super(SemAlignHead, self).__init__(
            in_channels=in_channels, **kwargs)
                
        def create_decoder():
            decoder_layer = TPN_DecoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim * 4)
            return TPN_Decoder(decoder_layer, num_layers)

        self.image_size = img_size
        self.use_stages = use_stages+1
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        self.unseen_idx = [idx for idx in all_idx if idx not in seen_idx]
        self.use_laa = use_laa
        self.laa_stages = laa_stages if use_laa else []
        self.text_dim = text_dim
        self.use_cata = use_cata
        self.use_efm = use_efm
        self.use_ccp = use_ccp

        nhead = num_heads
        dim = embed_dims

        self.cascade_decoders = nn.ModuleList([create_decoder() for _ in range(3)])

        if hasattr(self, 'conv_seg'):
            delattr(self, 'conv_seg')

        self.q_projs = nn.ModuleList([nn.Linear(dim * 2, dim) for _ in range(3)])
        
        if self.use_laa:
            self.laa_module = LAA(
                dim=embed_dims,
                qk_dim=embed_dims//2,
                mlp_dim=embed_dims,
                heads=laa_heads,
                ps=laa_patch_size,
                text_dim=text_dim,
                use_cata=use_cata,
                use_efm=use_efm,
                cata_lite=cata_lite
            )

        if self.use_ccp:
            self.cost_refiner = CostRefiner(in_channels=1)
            
            self.cost_feat_adapter = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dims // 4, embed_dims, 1)
            )
            
            self.ccp_feature_gate = nn.Parameter(torch.tensor(0.0))

        self.weights_s1 = nn.Parameter(gaussian(torch.tensor([2, 1, 0], device='cuda'), 0, 1))
        self.weights_s2 = nn.Parameter(gaussian(torch.tensor([2, 1, 0], device='cuda'), 0, 1))

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, self_training=False, st_mask=None):
        seg_logits = self.forward(inputs)

        if self_training:
            pseudo_semantic_masks = seg_logits['pred_masks'].clone().detach().sigmoid()
            pseudo_semantic_masks[:, self.seen_idx, :, :] = -1
            pseudo_semantic_seg = pseudo_semantic_masks.argmax(dim=1).unsqueeze(1)
            gt_semantic_seg[gt_semantic_seg==-1] = pseudo_semantic_seg[gt_semantic_seg==-1]
            gt_semantic_seg[gt_semantic_seg==-1] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)
        else:
            gt_semantic_seg[gt_semantic_seg==-1] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg, self_training):
        return self.forward(inputs, self_training)

    def decode(self, q, decoder, laterals, decoder_output):
        _, attn_ = decoder(q, laterals.transpose(0, 1))
        for attn in attn_:
            attn = attn.transpose(-1, -2)
            attn = self.d3_to_d4(attn)
            decoder_output.append(attn)
        return decoder_output

    def forward(self, inputs_both, self_training=None):
        inputs = inputs_both[0][0]
        cls_token = inputs_both[0][1]
        text_token = inputs_both[1]
        
        self._text_token = text_token

        processed_inputs = []
        for i, stage in enumerate(inputs[:self.use_stages]):
            if self.use_laa and i in self.laa_stages:
                try:
                    enhanced_stage = self.laa_module(stage, stage_idx=i, text_token=text_token)
                    processed_inputs.append(enhanced_stage)
                except Exception as e:
                    raise RuntimeError(f"LAA processing failed for stage {i}") from e
            else:
                processed_inputs.append(stage)

        if self.use_ccp:
            prev_cost = None
            ccp_enhanced_inputs = []
            
            
            for stage_feat in processed_inputs:
                B, C, H, W = stage_feat.shape
                
                if text_token is None:
                    ccp_enhanced_inputs.append(stage_feat)
                    continue
                    
                feat_norm = F.normalize(stage_feat, dim=1)
                
                txt_norm = F.normalize(text_token, dim=1)
                
                txt_norm = txt_norm.to(dtype=feat_norm.dtype)
                
                raw_cost = torch.einsum('bchw,nc->bnhw', feat_norm, txt_norm)
                
                current_cost = raw_cost
                if prev_cost is not None:
                    prev_cost_up = F.interpolate(prev_cost, size=(H, W), mode='bilinear', align_corners=False)
                    current_cost = current_cost + prev_cost_up
                
                B_N_H_W = current_cost.shape
                cost_reshaped = current_cost.reshape(-1, 1, H, W)
                refined_cost_reshaped = self.cost_refiner(cost_reshaped)
                refined_cost = refined_cost_reshaped.reshape(B_N_H_W)
                
                
                class_max_response = refined_cost.flatten(2).max(dim=2)[0]
                
                K = min(20, class_max_response.size(1))
                topk_values, topk_indices = torch.topk(class_max_response, K, dim=1)
                
                mask = torch.zeros_like(class_max_response)
                mask.scatter_(1, topk_indices, 1.0)
                mask = mask.unsqueeze(-1).unsqueeze(-1)
                
                masked_cost = refined_cost * mask
                
                cost_feature_raw = torch.einsum('bnhw,nc->bchw', masked_cost, txt_norm)
                
                cost_feature = self.cost_feat_adapter(cost_feature_raw)
                
                enhanced_feat = stage_feat + self.ccp_feature_gate.to(dtype=stage_feat.dtype) * cost_feature
                
                ccp_enhanced_inputs.append(enhanced_feat)
                
                prev_cost = refined_cost
            
            processed_inputs = ccp_enhanced_inputs
        
        laterals = [self.d4_to_d3(stage) if stage.dim() > 3 else stage for stage in processed_inputs]

        q_s1, q_s2, q_s3 = [self.q_projs[i](self.get_qs(text_token, cls_token)).transpose(0, 1) for i in range(3)]
  
        s1, s2 = laterals[:3], laterals[3:6]
        
        decoder_s1 = self.decode(q_s1, self.cascade_decoders[0], sum(w * s for w, s in zip(self.weights_s1, s1)), [])
        decoder_s2 = self.decode(q_s2, self.cascade_decoders[1], sum(w * s for w, s in zip(self.weights_s2, s2)), [])
        decoder_s3 = self.decode(q_s3, self.cascade_decoders[2], laterals[-1], [])

        out = {}
        out["pred_masks"] = F.interpolate((decoder_s1[-1]+decoder_s2[-1]+decoder_s3[-1]),size=(self.image_size, self.image_size),
                                          mode='bilinear', align_corners=False)
 
        if not self.training:
            if self_training:
                out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx)
            else:
                out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.1)

        return out

    def semantic_inference(self, mask_pred, seen_idx, weight=0.0):
        mask_pred = mask_pred.sigmoid()
        mask_pred[:,seen_idx] = mask_pred[:,seen_idx] - weight
        return mask_pred

    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        return [
            {"pred_masks": a}
            for a in outputs_seg_masks[:-1]
        ]

    def d3_to_d4(self, t):
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)

    def get_qs(self, q, cls):
        C, dim = q.shape
        bs, _ = cls.shape
        q = q.expand(bs, -1, -1)
        q1 = torch.einsum("bd,bcd->bcd", cls, q)
        q_ = torch.concat((q1, q), dim=-1)
        return q_

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label, num_classes=None):
        """Compute segmentation loss."""
        if isinstance(seg_logit, dict):
            seg_label = seg_label.squeeze(1)
            
            text_embeddings = getattr(self, '_text_token', None)
            
            try:
                loss = self.loss_decode(
                    seg_logit,
                    seg_label,
                    text_embeddings=text_embeddings,
                    ignore_index=self.ignore_index)
            except TypeError:
                loss = self.loss_decode(
                    seg_logit,
                    seg_label,
                    ignore_index=self.ignore_index)

            loss['acc_seg'] = accuracy(seg_logit["pred_masks"], seg_label, ignore_index=self.ignore_index)
            return loss
