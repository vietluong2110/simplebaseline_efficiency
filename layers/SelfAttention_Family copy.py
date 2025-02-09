import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange

import pywt
import math

import sys
sys.path.append('./clifford-group-equivariant-neural-networks')
from algebra.cliffordalgebra import CliffordAlgebra

import pdb


class WaveletEmbedding(nn.Module):
    def __init__(self, d_model=16, swt=True, requires_grad=False, wv='db2', m=2,):
        super().__init__()

        self.swt = swt
        self.d_model = d_model
        
        self.wavelet = pywt.Wavelet(wv)
        if self.swt:
            h0 = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
            h1 = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
        else:
            h0 = torch.tensor(self.wavelet.rec_lo[::-1], dtype=torch.float32)
            h1 = torch.tensor(self.wavelet.rec_hi[::-1], dtype=torch.float32)

        self.h0 = nn.Parameter(torch.tile(h0[None, None, :], [self.d_model, 1, 1]), requires_grad=requires_grad)
        self.h1 = nn.Parameter(torch.tile(h1[None, None, :], [self.d_model, 1, 1]), requires_grad=requires_grad)
        
        self.kernel_size = self.h0.shape[-1]
        self.m = m  # Number of decomposition levels

    def forward(self, x):
        if self.swt:
            coeffs = self.swt_decomposition(x, self.h0, self.h1, self.m, self.kernel_size)
        else:
            coeffs = self.swt_reconstruction(x, self.h0, self.h1, self.m, self.kernel_size)
        return coeffs

    def swt_decomposition(self, x, h0, h1, depth, kernel_size):
        approx_coeffs = x
        coeffs = []
        dilation = 1
        # pdb.set_trace()
        for _ in range(depth):
            padding = dilation * (kernel_size - 1)
            padding_r = (kernel_size * dilation) // 2
            pad = (padding - padding_r, padding_r)
            approx_coeffs_pad = F.pad(approx_coeffs, pad, "circular")
            detail_coeff = F.conv1d(approx_coeffs_pad, h1, dilation=dilation, groups=x.shape[1])
            approx_coeffs = F.conv1d(approx_coeffs_pad, h0, dilation=dilation, groups=x.shape[1])
            coeffs.append(detail_coeff)
            dilation *= 2
        coeffs.append(approx_coeffs)

        return torch.stack(list(reversed(coeffs)), -2)

    def swt_reconstruction(self, coeffs, g0, g1, m, kernel_size):
        # m = coeffs.shape[-2] - 1
        dilation = 2 ** (m - 1)
        approx_coeff = coeffs[:,:,0,:]
        detail_coeffs = coeffs[:,:,1:,:]
        
        # pdb.set_trace()
        for i in range(m):
            detail_coeff = detail_coeffs[:,:,i,:]
            padding = dilation * (kernel_size - 1)
            padding_l = (dilation * kernel_size) // 2
            pad = (padding_l, padding - padding_l)
            approx_coeff_pad = F.pad(approx_coeff, pad, "circular")
            detail_coeff_pad = F.pad(detail_coeff, pad, "circular")
            
            y = F.conv1d(approx_coeff_pad, g0, groups=approx_coeff.shape[1], dilation=dilation) + \
                F.conv1d(detail_coeff_pad, g1, groups=detail_coeff.shape[1], dilation=dilation)
            approx_coeff = y / 2
            dilation //= 2
            
        return approx_coeff

class WaveAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, d_head, requires_grad=True, wv='db2', m=2, p=0.5):
        super(WaveAttentionLayer, self).__init__()

        self.d_head = d_head
        self.inner_attention = attention
        self.d_model = d_model

        self.swt = WaveletEmbedding(d_model=self.d_head, swt=True, requires_grad=requires_grad, wv=wv, m=m)
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p)
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            WaveletEmbedding(d_model=self.d_head, swt=False, requires_grad=requires_grad, wv=wv, m=m),
        )

        d_metric = int(np.log2(self.d_head)) + 1
        metric = [1]*d_metric
        self.metric = torch.tensor(metric, dtype=torch.float)
        self.ca = CliffordAlgebra(metric)
        self.padding = (0, self.ca.n_blades-self.d_head)
        self.dropout = nn.Dropout(0.5)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        # Apply SWT
        queries = self.swt(queries)
        keys = self.swt(keys)
        values = self.swt(values)

        # Apply projections
        queries = self.dropout(self.query_projection(queries))
        keys = self.dropout(self.key_projection(keys))
        values = self.dropout(self.value_projection(values))
        
        # out_channel, _ = self.inner_attention(queries, keys, values)

        queries = queries.transpose(1,3)
        keys = keys.transpose(1,3)
        values = values.transpose(1,3)

        # out_token, _ = self.inner_attention(queries, keys, values)

        # out = out_channel.transpose(1,3) #+ out_token

        # pdb.set_trace()
        # out_ca = self.ca_projection(out)
        queries_ca = F.pad(queries, self.padding, mode='constant', value=0)
        queries_ca = self.ca.geometric_product(queries_ca, queries_ca)
        queries = queries + queries_ca[:,:,:,:self.d_head]

        keys_ca = F.pad(keys, self.padding, mode='constant', value=0)
        keys_ca = self.ca.geometric_product(keys_ca, keys_ca)
        keys = keys + keys_ca[:,:,:,:self.d_head]

        values_ca = F.pad(values, self.padding, mode='constant', value=0)
        values_ca = self.ca.geometric_product(values_ca, values_ca)
        values = values + values_ca[:,:,:,:self.d_head]

        out, _ = self.inner_attention(queries, keys, values)

        # Apply output projection
        out = self.out_projection(out.transpose(1,3))

        return out, None



# class WaveAttentionLayer(nn.Module):
#     def __init__(self, attention, d_model,
#                  requires_grad=True, wv='db2', m=2, d_head=None, p=0.5): #sym5
#         super(WaveAttentionLayer, self).__init__()

#         self.d_head = d_head
#         self.inner_attention = attention
#         self.d_model = d_model
        

#         self.swt = WaveletEmbedding(d_model=self.d_head, swt=True, requires_grad=requires_grad, wv=wv, m=m)
#         self.query_projection = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.Dropout(p)
#         )
#         self.key_projection = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.Dropout(p)
#         )
#         self.value_projection = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.Dropout(p)
#         )
#         self.out_projection = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             WaveletEmbedding(d_model=self.d_head, swt=False,
#                               requires_grad=requires_grad, wv=wv, m=m),
#         )
        
#         # metric = torch.tensor([1.0, 1.0, 1.0])
#         # self.ca = CliffordAlgebra(metric)


#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         # pdb.set_trace()
#         queries = self.swt(queries)
#         keys = self.swt(keys)
#         values = self.swt(values)

#         queries = self.query_projection(queries).permute(0,3,2,1)
#         keys = self.key_projection(keys).permute(0,3,2,1)
#         values = self.value_projection(values).permute(0,3,2,1)

#         wedge1 = self._wedge_product(keys, values)
#         wedge2 = self._wedge_product(keys.transpose(-1,-3), values.transpose(-1,-3)).transpose(-1,-3)
#         out = queries + wedge1 + wedge2

#         # # pdb.set_trace()
#         # blades = torch.tensor(range(self.d_head))  # Use only the first 7 blades
#         # gp1 = self.ca.geometric_product(keys, values, blades=(blades, blades, blades))
#         # # blades = torch.tensor(range(self.d_model))  # Use only the first 7 blades
#         # # gp2 = self.ca.geometric_product(keys.transpose(-1,-3), values.transpose(-1,-3), blades=(blades, blades, blades)).transpose(-1,-3)
#         # out = queries + gp1 # + gp2

#         # out, attn = self.inner_attention(
#         #     wedge,
#         #     keys,
#         #     values,
#         # )

#         out = self.out_projection(out.permute(0,3,2,1))

#         return out, None
    

#     def _wedge_product(self, keys, values):
#         assert keys.shape == values.shape, "Input tensors must have the same shape"
#         assert keys.shape[-1] >= 2, "The last dimension must be at least 2"

#         # pdb.set_trace()
#         wedge = torch.einsum('...i,...j->...ij', keys, values) - torch.einsum('...i,...j->...ji', keys, values)
        
#         # Extract the upper triangular part (excluding diagonal)
#         # embedding_dim = keys.shape[-1]
#         # mask = torch.triu(torch.ones(embedding_dim, embedding_dim), diagonal=1).bool()
#         # wedge = wedge[..., mask]
        
#         return wedge.mean(-1) # reduce dimension

#     def _geometric_product(self, a, b):
#         # Implement blade-based geometric product
#         blades = []
        
#         # Scalar part (grade 0)
#         scalar = torch.sum(a * b, dim=-1, keepdim=True)
#         blades.append(scalar)
        
#         # Vector part (grade 1)
#         vector = a + b
#         blades.append(vector)
        
#         # Higher grade blades
#         for grade in range(2, self.max_grade + 1):
#             blade = self._compute_blade(a, b, grade)
#             blades.append(blade)
        
#         # Concatenate all blades
#         result = torch.cat(blades, dim=-1)
        
#         return result
    
#     def _compute_blade(self, a, b, grade):
#         # Compute blade of given grade
#         combinations = torch.combinations(torch.arange(a.shape[-1]), r=grade)
#         blade = torch.zeros_like(a[..., :combinations.shape[0]])
        
#         for i, combo in enumerate(combinations):
#             term = torch.ones_like(a[..., 0])
#             for j in combo:
#                 term = term * (a[..., j] * b[..., j])
#             blade[..., i] = term
        
#         return blade


# class WaveletEmbedding(nn.Module):
#     def __init__(self, d_model=16, swt=True, requires_grad=False, wv='db2', m=2,):
#         super().__init__()

#         self.swt = swt
#         self.d_model = d_model
        
#         # pdb.set_trace()
#         self.wavelet = pywt.Wavelet(wv)
#         if self.swt:
#             h0 = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
#             h1 = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
#         else:
#             h0 = torch.tensor(self.wavelet.rec_lo[::-1], dtype=torch.float32)
#             h1 = torch.tensor(self.wavelet.rec_hi[::-1], dtype=torch.float32)

#         self.h0 = nn.Parameter(torch.tile(h0[None, None, :], [self.d_model, 1, 1]), requires_grad=requires_grad)
#         self.h1 = nn.Parameter(torch.tile(h1[None, None, :], [self.d_model, 1, 1]), requires_grad=requires_grad)
        
#         self.kernel_size = len(self.wavelet.dec_lo[::-1])
#         self.m = m  # Number of decomposition levels

#     def forward(self, x):
#         if self.swt:
#             coeffs = self.swt_decomposition(x, self.h0, self.h1, self.m, self.kernel_size)
#         else:
#             coeffs = self.swt_reconstruction(x, self.h0, self.h1, self.m, self.kernel_size)
#         return coeffs

#     def swt_decomposition(self, x, h0, h1, depth, kernel_size):
#         approx_coeffs = x
#         coeffs = []
#         dilation = 1
#         # pdb.set_trace()
#         for _ in range(depth):
#             padding = dilation * (kernel_size - 1)
#             padding_r = (kernel_size * dilation) // 2
#             pad = (padding - padding_r, padding_r)
#             approx_coeffs_pad = F.pad(approx_coeffs, pad, "circular")
#             detail_coeff = F.conv1d(approx_coeffs_pad, h1, dilation=dilation, groups=x.shape[1])
#             approx_coeffs = F.conv1d(approx_coeffs_pad, h0, dilation=dilation, groups=x.shape[1])
#             coeffs.append(detail_coeff)
#             dilation *= 2
#         coeffs.append(approx_coeffs)

#         return torch.stack(list(reversed(coeffs)), -2)

#     def swt_reconstruction(self, coeffs, g0, g1, m, kernel_size):
#         # m = coeffs.shape[-2] - 1
#         dilation = 2 ** (m - 1)
#         approx_coeff = coeffs[:,:,0,:]
#         detail_coeffs = coeffs[:,:,1:,:]
        
#         # pdb.set_trace()
#         for i in range(m):
#             detail_coeff = detail_coeffs[:,:,i,:]
#             padding = dilation * (kernel_size - 1)
#             padding_l = (dilation * kernel_size) // 2
#             pad = (padding_l, padding - padding_l)
#             approx_coeff_pad = F.pad(approx_coeff, pad, "circular")
#             detail_coeff_pad = F.pad(detail_coeff, pad, "circular")
            
#             y = F.conv1d(approx_coeff_pad, g0, groups=approx_coeff.shape[1], dilation=dilation) + \
#                 F.conv1d(detail_coeff_pad, g1, groups=detail_coeff.shape[1], dilation=dilation)
#             approx_coeff = y / 2
#             dilation //= 2
            
#         return approx_coeff


# class WaveAttentionLayer(nn.Module):
#     def __init__(self, attention, d_model,
#                  requires_grad=True, wv='db2', m=2, d_head=None, p=0.5): #sym5
#         super(WaveAttentionLayer, self).__init__()

#         self.d_head = d_head
#         self.inner_attention = attention
        

#         self.swt = WaveletEmbedding(d_model=self.d_head, swt=True, requires_grad=requires_grad, wv=wv, m=m)
#         self.query_projection = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.Dropout(p)
#         )
#         self.key_projection = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.Dropout(p)
#         )
#         self.value_projection = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.Dropout(p)
#         )
#         self.out_projection = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             WaveletEmbedding(d_model=self.d_head, swt=False,
#                               requires_grad=requires_grad, wv=wv, m=m),
#         )

#         self.clifford = create_clifford_algebra(self.d_head)
        
#     def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
#         # pdb.set_trace()
#         queries = self.swt(queries)
#         keys = self.swt(keys)
#         values = self.swt(values)

#         queries = self.query_projection(queries).permute(0,3,2,1)
#         keys = self.key_projection(keys).permute(0,3,2,1)
#         values = self.value_projection(values).permute(0,3,2,1)

#         pdb.set_trace()
#         clifford_interaction = self.clifford.geometric_product(keys, values)
        
#         wedge1 = self._wedge_product(keys, values)
#         wedge2 = self._wedge_product(keys.transpose(-1,-3), values.transpose(-1,-3)).transpose(-1,-3)
#         out = queries + wedge1 + wedge2


#         # out, attn = self.inner_attention(
#         #     wedge,
#         #     keys,
#         #     values,
#         # )

#         out = self.out_projection(out.permute(0,3,2,1))

#         return out, None
    

#     def _wedge_product(self, keys, values):
#         assert keys.shape == values.shape, "Input tensors must have the same shape"
#         assert keys.shape[-1] >= 2, "The last dimension must be at least 2"

#         # pdb.set_trace()
#         wedge = torch.einsum('...i,...j->...ij', keys, values) - torch.einsum('...i,...j->...ji', keys, values)
        
#         # Extract the upper triangular part (excluding diagonal)
#         # embedding_dim = keys.shape[-1]
#         # mask = torch.triu(torch.ones(embedding_dim, embedding_dim), diagonal=1).bool()
#         # wedge = wedge[..., mask]
        
#         return wedge.mean(-1) # reduce dimension




# Code implementation from https://github.com/thuml/Flowformer
class FlowAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(FlowAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # kernel
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        # incoming and outgoing
        normalizer_row = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + 1e-6, keys.sum(dim=2) + 1e-6))
        normalizer_col = 1.0 / (torch.einsum("nhsd,nhd->nhs", keys + 1e-6, queries.sum(dim=2) + 1e-6))
        # reweighting
        normalizer_row_refine = (
            torch.einsum("nhld,nhd->nhl", queries + 1e-6, (keys * normalizer_col[:, :, :, None]).sum(dim=2) + 1e-6))
        normalizer_col_refine = (
            torch.einsum("nhsd,nhd->nhs", keys + 1e-6, (queries * normalizer_row[:, :, :, None]).sum(dim=2) + 1e-6))
        # competition and allocation
        normalizer_row_refine = torch.sigmoid(
            normalizer_row_refine * (float(queries.shape[2]) / float(keys.shape[2])))
        normalizer_col_refine = torch.softmax(normalizer_col_refine, dim=-1) * keys.shape[2]  # B h L vis
        # multiply
        kv = keys.transpose(-2, -1) @ (values * normalizer_col_refine[:, :, :, None])
        x = (((queries @ kv) * normalizer_row[:, :, :, None]) * normalizer_row_refine[:, :, :, None]).transpose(1,
                                                                                                                2).contiguous()
        return x, None


# Code implementation from https://github.com/shreyansh26/FlashAttention-PyTorch
class FlashAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FlashAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def flash_attention_forward(self, Q, K, V, mask=None):
        BLOCK_SIZE = 32
        NEG_INF = -1e10  # -infinity
        EPSILON = 1e-10
        # mask = torch.randint(0, 2, (128, 8)).to(device='cuda')
        O = torch.zeros_like(Q, requires_grad=True)
        l = torch.zeros(Q.shape[:-1])[..., None]
        m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF

        O = O.to(device='cuda')
        l = l.to(device='cuda')
        m = m.to(device='cuda')

        Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[-1])
        KV_BLOCK_SIZE = BLOCK_SIZE

        Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
        K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
        V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
        if mask is not None:
            mask_BLOCKS = list(torch.split(mask, KV_BLOCK_SIZE, dim=1))

        Tr = len(Q_BLOCKS)
        Tc = len(K_BLOCKS)

        O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
        l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
        m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

        for j in range(Tc):
            Kj = K_BLOCKS[j]
            Vj = V_BLOCKS[j]
            if mask is not None:
                maskj = mask_BLOCKS[j]

            for i in range(Tr):
                Qi = Q_BLOCKS[i]
                Oi = O_BLOCKS[i]
                li = l_BLOCKS[i]
                mi = m_BLOCKS[i]

                scale = 1 / np.sqrt(Q.shape[-1])
                Qi_scaled = Qi * scale

                S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi_scaled, Kj)
                if mask is not None:
                    # Masking
                    maskj_temp = rearrange(maskj, 'b j -> b 1 1 j')
                    S_ij = torch.where(maskj_temp > 0, S_ij, NEG_INF)

                m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
                P_ij = torch.exp(S_ij - m_block_ij)
                if mask is not None:
                    # Masking
                    P_ij = torch.where(maskj_temp > 0, P_ij, 0.)

                l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON

                P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)

                mi_new = torch.maximum(m_block_ij, mi)
                li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij

                O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi + (
                        torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
                l_BLOCKS[i] = li_new
                m_BLOCKS[i] = mi_new

        O = torch.cat(O_BLOCKS, dim=2)
        l = torch.cat(l_BLOCKS, dim=2)
        m = torch.cat(m_BLOCKS, dim=2)
        return O, l, m

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        res = \
        self.flash_attention_forward(queries.permute(0, 2, 1, 3), keys.permute(0, 2, 1, 3), values.permute(0, 2, 1, 3),
                                     attn_mask)[0]
        return res.permute(0, 2, 1, 3).contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        # pdb.set_trace()
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        mask = torch.tril(torch.ones(L, L)).to(scores.device)
        scores = torch.einsum("bhls,ls->bhls", scores, mask)
        
        # if self.mask_flag:
        #     if attn_mask is None:
        #         attn_mask = TriangularCausalMask(B, L, device=queries.device)
        #     scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


# Code implementation from https://github.com/zhouhaoyi/Informer2020
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # pdb.set_trace()
        queries = self.query_projection(queries).view(B, L, H, -1) # [16, 866, 512] -> [16, 866, 8, 64]
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None

