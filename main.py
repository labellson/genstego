import cv2
import numpy as np

from bitarray import bitarray
from scanner import MatScanner
from embedder import Embedder

def embed(stego, secret, chromosome):
    # Convert to a flattened pixel sequence
    stego = MatScanner.scan(stego, chromosome[2], chromosome[1], chromosome[0])
    secret = secret.flatten()
    return Embedder.embed(stego, secret, chromosome)

def fitness(stego, secret, chromosome):
    # Embed the secret sequence
    try:
        stego = embed(stego, secret, chromosome)
    except:
        return 0
    pass

# Inputs
stego = np.arange(9).reshape(3,3)
secret = np.array([5, 3])

# Scan using raster order
stego_sequence = MatScanner.scan(stego, 0, 0, MatScanner.Direction.raster)

# Embed secret pixels
stego_embedded = Embedder.embed(stego_sequence, secret, np.array([0, 0, 0, 3, 1, 1, 1]))

print('- Stego: \n{}'.format(stego))
print('\n- Embedded: {}'.format(stego_embedded))

