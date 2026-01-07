
import torch
import torch.nn.functional as F

def test_unfold_shape():
    B = 1
    dim = 2
    H, W = 4, 4
    N = H * W
    
    # Create a feature map where we know the values
    # (B, C, H, W)
    # Channel 0: 0..15
    # Channel 1: 16..31
    feat = torch.arange(N, dtype=torch.float32).reshape(1, 1, H, W).repeat(1, dim, 1, 1)
    feat[:, 1, :, :] += N
    
    # User's logic simulation
    output_feats = feat.flatten(2).transpose(1, 2) # (B, N, C)
    
    window_size = 3
    pad = (window_size - 1) // 2
    win_pixels = window_size ** 2
    
    output_feat_map = output_feats.transpose(1, 2).reshape(B, dim, H, W)
    
    output_windows = F.unfold(
        F.pad(output_feat_map, (pad, pad, pad, pad)),
        kernel_size=window_size,
        stride=1
    ) # (B, dim*win_pixels, N)
    
    print(f"Unfold shape: {output_windows.shape}")
    
    # Check reshape
    reshaped = output_windows.reshape(B, dim, win_pixels, N)
    print(f"Reshaped shape: {reshaped.shape}")
    
    # Check content of the first window (top-left at 0,0)
    # Center is (0,0). Padded. 
    # Window covers (-1,-1) to (1,1).
    # Inside image: (0,0), (0,1), (1,0), (1,1). Rest are padded (0).
    
    # Let's check reshaped[0, 0, :, 0] -> Channel 0, Window 0 (center 0,0)
    # The window flattened.
    w0_c0 = reshaped[0, 0, :, 0]
    print(f"Window 0 Channel 0: {w0_c0}")
    
    # Check if reshape splits dimensions correctly.
    # Unfold output is:
    # Row 0: Ch0, k0,0
    # Row 1: Ch0, k0,1
    # ...
    # Row 9: Ch0, k2,2
    # Row 10: Ch1, k0,0 ...
    
    # If standard PyTorch unfold is used, the order is (Channel, Kernel_H, Kernel_W).
    # So the first 9 elements are Channel 0. The next 9 are Channel 1.
    # So reshaped(B, dim, win_pixels, N) splits the first dimension (dim*win_pixels) into (dim, win_pixels).
    # Since reshape fills row-major (last dim fastest),
    # If the underlying data is C_slow, K_fast.
    # Then reshape(C, K) works.
    # Let's verify if unfold produces C_slow, K_fast.
    pass

test_unfold_shape()
