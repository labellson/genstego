import cv2
import numpy as np

from bitarray import bitarray
from scanner import MatScanner
from embedder import Embedder

# Inputs
stego = np.arange(9).reshape(3,3)
secret = np.array([5, 3])

# Scan using raster order
stego_sequence = MatScanner.scan(stego, 0, 0, MatScanner.Direction.raster)

# Embed secret pixels
stego_embedded = Embedder.embed(stego_sequence, secret, np.array([1, 1, 1, 1]))

print('- Stego: \n{}'.format(stego))
print('\n- Embedded: {}'.format(stego_embedded))
