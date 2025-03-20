import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
import pywt



class WaveletEmbedding(nn.Module):
    def __init__(self, d_channel=16, swt=True, requires_grad=False, wv='db2', m=2,
                 kernel_size=None):
        super().__init__()

        self.swt = swt
        self.d_channel = d_channel
        self.m = m  # Number of decomposition levels of detailed coefficients
        
        if kernel_size is None:
            self.wavelet = pywt.Wavelet(wv)
            if self.swt:
                h0 = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
            else:
                h0 = torch.tensor(self.wavelet.rec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.rec_hi[::-1], dtype=torch.float32)
            self.h0 = nn.Parameter(torch.tile(h0[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.tile(h1[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            self.kernel_size = self.h0.shape[-1]
        else:
            self.kernel_size = kernel_size
            self.h0 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)
            nn.init.xavier_uniform_(self.h0)
            nn.init.xavier_uniform_(self.h1)
        
            with torch.no_grad():
                self.h0.data = self.h0.data / torch.norm(self.h0.data, dim=-1, keepdim=True)
                self.h1.data = self.h1.data / torch.norm(self.h1.data, dim=-1, keepdim=True)


    def forward(self, x):
        if self.swt:
            coeffs = self.swt_decomposition(x, self.h0, self.h1, self.m, self.kernel_size)
        else:
            coeffs = self.swt_reconstruction(x, self.h0, self.h1, self.m, self.kernel_size)
        return coeffs

    def upsample_filter_torch(self, h, factor):
        """
        Upsample a 1D filter h by inserting (factor - 1) zeros between coefficients.
        h is a 3D tensor of shape [C_out, C_in, kernel_size].
        Returns a tensor of shape [C_out, C_in, (kernel_size-1)*factor + 1].
        """
        C_out, C_in, kernel_size = h.shape
        up_kernel_size = (kernel_size - 1) * factor + 1
        h_upsampled = torch.zeros(C_out, C_in, up_kernel_size, device=h.device, dtype=h.dtype)
        h_upsampled[..., ::factor] = h
        return h_upsampled

    def fft_convolve_1d(self, x, filt):
        """
        Compute circular convolution of x with filt using FFT.
        
        Parameters:
        x: Tensor of shape [B, C, L] (input signal)
        filt: Tensor of shape [C, 1, L_f] (filter)
                Assumes that each channel in x is convolved with its corresponding filter.
        
        The convolution is performed circularly (with periodic extension), and the output length is L.
        
        Returns:
        y: Tensor of shape [B, C, L]
        """
        B, C, L = x.shape
        L_f = filt.shape[-1]
        fft_length = L  # We use L as the FFT length for circular convolution.

        # Compute the FFT of the input signal x along the last dimension.
        X_fft = torch.fft.rfft(x, n=fft_length)  # Shape: [B, C, fft_length//2 + 1]

        # Compute the FFT of the filter. filt has shape [C, 1, L_f],
        # so after FFT it has shape [C, 1, fft_length//2 + 1].
        F_fft = torch.fft.rfft(filt, n=fft_length)  # Shape: [C, 1, fft_length//2 + 1]

        # Since we assume the filter is applied independently per channel,
        # squeeze the singleton dimension so that F_fft becomes [C, fft_length//2 + 1].
        F_fft = F_fft.squeeze(1)  # Shape: [C, fft_length//2 + 1]

        # Add a batch dimension for broadcasting (so it becomes [1, C, fft_length//2 + 1]).
        F_fft = F_fft.unsqueeze(0)

        # Now multiply in the frequency domain.
        # X_fft has shape [B, C, fft_length//2 + 1] and F_fft is broadcasted to [B, C, fft_length//2 + 1].
        Y_fft = X_fft * F_fft

        # Compute the inverse FFT to return to the time domain.
        y = torch.fft.irfft(Y_fft, n=fft_length)  # Shape: [B, C, fft_length]

        return y

    def swt_decomposition(self, x, h0, h1, depth, kernel_size):
        """
        Compute the SWT decomposition using FFT-based convolution.
        
        According to the professor's derivation, the detail coefficients can be computed as:
        d_n = (h_1 ↑ 2^(n-1)) * x
        where ↑ denotes upsampling.
        
        Parameters:
        x         : Input signal tensor of shape [batch, channels, length]
        h0        : Base low-pass filter as a tensor of shape [C, C, kernel_size]
        h1        : Base high-pass filter as a tensor of shape [C, C, kernel_size]
        depth     : Number of decomposition levels.
        kernel_size: Size of the base filters.
        
        Returns:
        A tensor containing the detail coefficients for each level (and final approximation)
        stacked along a new dimension. Levels are returned in reverse order.
        """
        coeffs = []
        dilation = 1
        # For each level, compute detail coefficients using FFT-based convolution.
        for n in range(1, depth + 1):
            factor = 2 ** (n - 1)
            # Upsample the high-pass filter: h1 ↑ 2^(n-1)
            h1_upsampled = self.upsample_filter_torch(h1, factor)
            detail_coeff = self.fft_convolve_1d(x, h1_upsampled)
            coeffs.append(detail_coeff)
        # For the final approximation, we can compute:
        factor = 2 ** (depth - 1)
        h0_upsampled = self.upsample_filter_torch(h0, factor)
        approx_coeff = self.fft_convolve_1d(x, h0_upsampled)
        coeffs.append(approx_coeff)
        
        # Optionally, reverse the list so that the coarsest scale (approximation) comes first.
        return torch.stack(coeffs[::-1], dim=-2)

    def swt_reconstruction(self, coeffs, g0, g1, m, kernel_size):
        dilation = 2 ** (m - 1)
        approx_coeff = coeffs[:,:,0,:]
        detail_coeffs = coeffs[:,:,1:,:]
        
        for i in range(m):
            detail_coeff = detail_coeffs[:,:,i,:]
            y = self.fft_convolve_1d(approx_coeff, g0) + self.fft_convolve_1d(detail_coeff, g1)
            approx_coeff = y / 2
            dilation //= 2
            
        return approx_coeff


class GeomAttentionLayer(nn.Module):
    def __init__(self, attention, d_model,
                 requires_grad=True, wv='db2', m=2, kernel_size=None,
                 d_channel=None, geomattn_dropout=0.5,): #sym5
        super(GeomAttentionLayer, self).__init__()

        self.d_channel = d_channel
        self.inner_attention = attention
        

        self.swt = WaveletEmbedding(d_channel=self.d_channel, swt=True, requires_grad=requires_grad, wv=wv, m=m, kernel_size=kernel_size)
        self.query_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.key_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.value_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            WaveletEmbedding(d_channel=self.d_channel, swt=False, requires_grad=requires_grad, wv=wv, m=m, kernel_size=kernel_size),
        )
        
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        queries = self.swt(queries)
        keys = self.swt(keys)
        values = self.swt(values)

        queries = self.query_projection(queries).permute(0,3,2,1)
        keys = self.key_projection(keys).permute(0,3,2,1)
        values = self.value_projection(values).permute(0,3,2,1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = self.out_projection(out.permute(0,3,2,1))


        return out, attn

class GeomAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, 
                 output_attention=False,
                 alpha=1.,):
        super(GeomAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        self.alpha = alpha 

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, _ = values.shape
        scale = self.scale or 1. / sqrt(E)

        dot_product = torch.einsum("blhe,bshe->bhls", queries, keys)

        queries_norm2 = torch.sum(queries**2, dim=-1)
        keys_norm2 = torch.sum(keys**2, dim=-1)
        queries_norm2 = queries_norm2.permute(0, 2, 1).unsqueeze(-1)  # (B, H, L, 1)
        keys_norm2 = keys_norm2.permute(0, 2, 1).unsqueeze(-2)        # (B, H, 1, S)
        wedge_norm2 = queries_norm2 * keys_norm2 - dot_product ** 2  # (B, H, L, S)
        wedge_norm2 = F.relu(wedge_norm2)
        wedge_norm = torch.sqrt(wedge_norm2 + 1e-8)

        scores = (1 - self.alpha) * dot_product + self.alpha * wedge_norm
        scores = scores * scale

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.tril(torch.ones(L, S)).to(scores.device)
            scores.masked_fill_(attn_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        A = self.dropout(torch.softmax(scores, dim=-1)) 

        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), scores.abs().mean())
        else:
            return (V.contiguous(), scores.abs().mean())

