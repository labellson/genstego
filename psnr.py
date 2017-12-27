import numpy as np
import math

def psnr(img1, img2):
    """Computes psnr fitness function"""
    mse = np.mean((img1 - img2)**2)

    if mse == 0:
        return 100
    return 10 * math.log10(255 / mse)
