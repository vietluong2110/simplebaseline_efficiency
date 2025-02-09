import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from math import sqrt
from typing import Optional, Tuple
from torch import Tensor
import torch.jit as jit

@torch.jit.script
def compute_attention_scores(queries: Tensor, keys: Tensor, scale: float, alpha: float) -> Tuple[Tensor, Tensor]:
    """JIT-compiled attention score computation for better performance"""
    dot_product = torch.matmul(queries, keys.transpose(-2, -1))
    
    if alpha == 0.0:
        return dot_product * scale, dot_product
    
    # Fused operation for norm computation
    queries_norm2 = torch.sum(queries.pow(2), dim=-1, keepdim=True)
    keys_norm2 = torch.sum(keys.pow(2), dim=-1, keepdim=True).transpose(-2, -1)
    
    # Compute wedge norm with better numerical stability
    wedge_norm2 = torch.clamp(queries_norm2 * keys_norm2 - dot_product.pow(2), min=1e-8)
    wedge_norm = torch.sqrt(wedge_norm2)
    
    if alpha == 1.0:
        scores = wedge_norm * scale
    else:
        scores = (1 - alpha) * dot_product + alpha * wedge_norm
        scores = scores * scale
    
    return scores, dot_product

class WaveletEmbedding(nn.Module):
    def __init__(self, d_channel=16, swt=True, requires_grad=False, wv='db2', m=2, kernel_size=None):
        super().__init__()
        self.swt = swt
        self.d_channel = d_channel
        self.m = m
        self._initialize_filters(kernel_size, wv, requires_grad)
        
        # Pre-compute padding values for each level
        self.padding_cache = []
        dilation = 1
        for _ in range(m):
            padding = dilation * (self.kernel_size - 1)
            padding_r = (self.kernel_size * dilation) // 2
            self.padding_cache.append((padding - padding_r, padding_r))
            dilation *= 2
        
        # Register buffers for faster GPU transfer
        self.register_buffer('dilation_factors', torch.tensor([2**i for i in range(m)]))
    
    @torch.jit.ignore
    def _initialize_filters(self, kernel_size, wv, requires_grad):
        if kernel_size is None:
            wavelet = pywt.Wavelet(wv)
            h_lo, h_hi = (wavelet.dec_lo[::-1], wavelet.dec_hi[::-1]) if self.swt else (wavelet.rec_lo[::-1], wavelet.rec_hi[::-1])
            h0 = torch.tensor(h_lo, dtype=torch.float32)
            h1 = torch.tensor(h_hi, dtype=torch.float32)
            
            # Use repeat instead of expand for contiguous memory
            self.h0 = nn.Parameter(h0[None, None, :].repeat(self.d_channel, 1, 1), requires_grad=requires_grad)
            self.h1 = nn.Parameter(h1[None, None, :].repeat(self.d_channel, 1, 1), requires_grad=requires_grad)
            self.kernel_size = self.h0.shape[-1]
        else:
            self.kernel_size = kernel_size
            self.h0 = nn.Parameter(torch.empty(self.d_channel, 1, kernel_size), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.empty(self.d_channel, 1, kernel_size), requires_grad=requires_grad)
            
            # Initialize filters with orthogonal initialization
            nn.init.orthogonal_(self.h0)
            nn.init.orthogonal_(self.h1)
            
            with torch.no_grad():
                self.h0.div_(self.h0.norm(dim=-1, keepdim=True))
                self.h1.div_(self.h1.norm(dim=-1, keepdim=True))

    def forward(self, x: Tensor) -> Tensor:
        return self.swt_decomposition(x) if self.swt else self.swt_reconstruction(x)

    def swt_decomposition(self, x: Tensor) -> Tensor:
        batch_size, channels, seq_len = x.shape
        coeffs = []
        approx_coeffs = x
        
        for level, (pad_left, pad_right) in enumerate(self.padding_cache):
            dilation = self.dilation_factors[level].item()
            
            # Single padding operation
            approx_coeffs_pad = F.pad(approx_coeffs, (pad_left, pad_right), "circular")
            
            # Parallel convolution operations with grouped convolutions
            detail_coeff = F.conv1d(approx_coeffs_pad, self.h1, dilation=dilation, groups=channels)
            approx_coeffs = F.conv1d(approx_coeffs_pad, self.h0, dilation=dilation, groups=channels)
            
            coeffs.append(detail_coeff)
        
        coeffs.append(approx_coeffs)
        return torch.stack(list(reversed(coeffs)), -2)

    def swt_reconstruction(self, coeffs):
        if len(coeffs.shape) != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {coeffs.shape}")
            
        batch_size, channels, levels, seq_len = coeffs.shape
        dilation = 2 ** (self.m - 1)
        approx_coeff = coeffs[:,:,0,:]
        detail_coeffs = coeffs[:,:,1:,:]
        
        for i in range(self.m):
            detail_coeff = detail_coeffs[:,:,i,:]
            padding = dilation * (self.kernel_size - 1)
            padding_l = (dilation * self.kernel_size) // 2
            pad = (padding_l, padding - padding_l)
            
            # Fused padding operation
            approx_coeff_pad = F.pad(approx_coeff, pad, "circular")
            detail_coeff_pad = F.pad(detail_coeff, pad, "circular")
            
            # Optimized convolution operations
            approx_coeff = (F.conv1d(approx_coeff_pad, self.h0, groups=channels, dilation=dilation) + 
                          F.conv1d(detail_coeff_pad, self.h1, groups=channels, dilation=dilation)) / 2
            
            dilation //= 2
            
        return approx_coeff

class GeomAttention(nn.Module):
    def __init__(self, mask_flag: bool = False, factor: int = 5, scale: Optional[float] = None,
                 attention_dropout: float = 0.1, output_attention: bool = False, alpha: float = 1.):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.alpha = alpha
        
        # Register causal mask as buffer for faster access
        self.register_buffer('cached_mask', None)
        
    def forward(self, queries: Tensor, keys: Tensor, values: Tensor,
                attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)
        
        # Efficient transpose operations
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        
        # Compute attention scores using JIT-compiled function
        scores, dot_product = compute_attention_scores(queries, keys, scale, self.alpha)
        
        if self.mask_flag:
            if attn_mask is None:
                if self.cached_mask is None or self.cached_mask.size(-1) != S:
                    self.cached_mask = torch.tril(torch.ones(L, S, device=queries.device))
                scores.masked_fill_(~self.cached_mask.unsqueeze(1), float('-inf'))
            else:
                scores.masked_fill_(~attn_mask.unsqueeze(1), float('-inf'))
        
        # Fused softmax and dropout
        A = self.dropout(F.softmax(scores, dim=-1))
        
        # Optimized output computation
        V = torch.einsum('bhls,bshd->blhd', A, values)
        
        return V.contiguous(), scores.abs().mean()

class GeomAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, requires_grad=True, wv='db2', m=2,
                 kernel_size=None, d_channel=None, geomattn_dropout=0.5):
        super().__init__()
        self.d_channel = d_channel
        self.inner_attention = attention
        
        self.swt = WaveletEmbedding(d_channel=self.d_channel, swt=True,
                                   requires_grad=requires_grad, wv=wv, m=m,
                                   kernel_size=kernel_size)
        
        # Use a single linear layer with larger output and split it
        self.combined_projection = nn.Linear(d_model, d_model * 3)
        self.dropout = nn.Dropout(geomattn_dropout)
        
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            WaveletEmbedding(d_channel=self.d_channel, swt=False,
                            requires_grad=requires_grad, wv=wv, m=m,
                            kernel_size=kernel_size),
        )
    
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        values = self.swt(values)
        
        # Single matrix multiplication for all projections
        combined = self.combined_projection(values)
        chunk_size = combined.size(-1) // 3
        queries, keys, values = combined.chunk(3, dim=-1)
        
        # Apply dropout to all projections at once
        combined = self.dropout(combined)
        queries, keys, values = combined.chunk(3, dim=-1)
        
        # Permute all tensors at once
        queries, keys, values = [x.permute(0, 3, 2, 1) for x in (queries, keys, values)]
        
        out, attn = self.inner_attention(queries, keys, values)
        out = self.out_projection(out.permute(0, 3, 2, 1))
        
        return out, attn


