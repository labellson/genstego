import cv2
import numpy as np

from bitarray import bitarray
from scanner import MatScanner
from embedder import Embedder
from psnr import psnr

def embed(stego, secret, chromosome):
    """Embed secret message into the host using the chromosome"""
    # Convert to a flattened pixel sequence
    stego = MatScanner.scan_genetic(stego, chromosome)
    secret = secret.flatten()
    return Embedder.embed(stego, secret, chromosome)

def fitness(stego, secret, chromosome):
    """Computes fitness for current chromosome"""
    # Embed the secret sequence
    try:
        stego_sequence = embed(stego, secret, chromosome)
    except:
        return 0

    # Reshape the stego image
    stego1 = MatScanner.reshape_genetic(stego_sequence, stego.shape, chromosome)
    return psnr(stego, stego1)


# Inputs
stego = np.arange(9).reshape(3,3)
secret = np.array([5, 3])
chromosome = np.array([0, 0, 0, 3, 0, 1, 1])

# Scan using raster order
stego_sequence = MatScanner.scan(stego, 0, 0, MatScanner.Direction.raster)

# Embed secret pixels
stego_embedded = Embedder.embed(stego_sequence, secret, chromosome)

stego_embedded = MatScanner.reshape(stego_embedded, stego.shape, chromosome[2], chromosome[1], chromosome[0])

print('- Stego: \n{}'.format(stego))
print('\n- Embedded: \n{}'.format(stego_embedded))
print('\n- Fitness: {}'.format(fitness(stego, secret, chromosome)))

