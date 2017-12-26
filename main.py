import cv2
import numpy as np

from bitarray import bitarray
from scanner import MatScanner

# Inputs
stego = np.arange(9).reshape(3,3)
secret = np.array([5])

# Scan using raster order
stego_sequence = MatScanner.scan(stego, 0, 0, MatScanner.Direction.raster)
# Unpack to bits
stego_bits = np.unpackbits(stego_sequence.reshape(-1,1).astype('uint8'), 1)
