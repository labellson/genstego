import numpy as np
import math

def psnr(img1, img2):
    """Computes psnr fitness function"""
    # Change the format of the matrix
    if img1.dtype != (np.float32 or np.float64):
        img1 = img1.astype(np.float32)

    if img2.dtype != (np.float32 or np.float64):
        img2 = img2.astype(np.float32)
    
    mse = np.mean((img1 - img2)**2)

    if mse == 0:
        return 100
    return 10 * math.log10(255 / mse)
