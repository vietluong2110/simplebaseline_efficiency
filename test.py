import torch
def upsample_filters_vectorized(h, depth):
    """
    Vectorized upsampling of filter h for multiple levels without an explicit Python loop.
    
    Parameters:
        h     : Tensor of shape [C, kernel_size]
        depth : Number of levels (for detail coefficients)
    
    Returns:
        upsampled_filters: Tensor of shape [depth, C, L_max] where L_max is the maximum upsampled length.
        indices_mask   : A tensor mask indicating valid positions (optional, if needed later)
    """
    import torch

    C, kernel_size = h.shape
    # Compute upsampling factors for each level: factors = [1, 2, 4, ..., 2^(depth-1)]
    factors = 2 ** torch.arange(0, depth, device=h.device, dtype=torch.long)  # shape: [depth]
    # Compute lengths for each level: L_level = (kernel_size - 1) * factor + 1
    L_levels = (kernel_size - 1) * factors + 1  # shape: [depth]
    L_max = L_levels.max().item()  # maximum length
    
    # Create a tensor to hold the upsampled filters: [depth, C, L_max]
    upsampled = torch.zeros(depth, C, L_max, device=h.device, dtype=h.dtype)
    
    # We'll create an index tensor for each level that indicates the positions where h's coefficients go.
    # We want a tensor of shape [depth, kernel_size] where for level i:
    #   indices[i] = [0, factor_i, 2*factor_i, ..., (kernel_size-1)*factor_i]
    indices = factors.unsqueeze(1) * torch.arange(0, kernel_size, device=h.device).unsqueeze(0)  # [depth, kernel_size]
    # Expand indices so that we can scatter into dimension 2: [depth, C, kernel_size]
    indices_expanded = indices.unsqueeze(1).expand(depth, C, kernel_size)
    
    # Now, expand h so that it can be scattered over levels: [depth, C, kernel_size]
    h_expanded = h.unsqueeze(0).expand(depth, C, kernel_size)
    
    # Scatter the coefficients into the appropriate positions along dimension 2.
    upsampled.scatter_(dim=2, index=indices_expanded, src=h_expanded)
    
    return upsampled, L_levels  # L_levels tells you the valid lengths for each level

# Example usage for h1:
# Assume h1 is of shape [C, 1, kernel_size]. We first squeeze the singleton dimension.
h1 = torch.tensor([[1, -1]])
h1_squeezed = h1.squeeze(1)  # shape: [C, kernel_size]
upsampled_h1, valid_lengths = upsample_filters_vectorized(h1_squeezed, 3)

print(upsampled_h1)