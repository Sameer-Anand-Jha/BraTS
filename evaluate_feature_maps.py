# evaluate_feature_maps.py

import torch
import torch.nn.functional as F
import numpy as np

def compute_correlation(tensor1_2d, tensor2_2d):
    """
    Compute Pearson correlation between two 2D feature maps of the same shape.
    Both inputs must be 2D (H, W). We flatten to 1D before computing correlation.
    """
    v1 = tensor1_2d.flatten().detach().cpu().numpy()
    v2 = tensor2_2d.flatten().detach().cpu().numpy()
    if v1.std() == 0 or v2.std() == 0:
        return 0.0
    return np.corrcoef(v1, v2)[0, 1]

def evaluate_similarity(model, input_tensor):
    """
    Runs a single forward pass through `model`, extracts:
      - `bottleneck`: shape (1, C_b, H_b, W_b)
      - `output`    : shape (1, C_o, H, W)

    Steps:
      1) Upsample bottleneck to (1, C_b, H, W)
      2) Mean over channels to get (1,1,H,W)
      3) If output has multiple channels, mean them to 1 channel
      4) Correlate the two 2D maps
    """
    model.eval()
    with torch.no_grad():
        output, bottleneck = model(input_tensor)
        # Upsample bottleneck to match output spatial size
        up_bottleneck = F.interpolate(
            bottleneck,
            size=output.shape[2:],
            mode='bilinear',
            align_corners=False
        )  # (1, C_b, H, W)

        # Collapse C_b channels by mean
        mean_bottleneck = up_bottleneck.mean(dim=1, keepdim=True)  # (1,1,H,W)

        # Collapse output channels if >1 (here typically C_o=1)
        if output.shape[1] > 1:
            mean_output = output.mean(dim=1, keepdim=True)
        else:
            mean_output = output  # (1,1,H,W)

        # Squeeze to 2D maps
        map_b = mean_bottleneck.squeeze(0).squeeze(0)  # (H,W)
        map_o = mean_output.squeeze(0).squeeze(0)      # (H,W)

        # Compute correlation
        corr = compute_correlation(map_b, map_o)
        return corr
