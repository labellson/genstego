import cv2
import numpy as np
import random

from scanner import MatScanner
from embedder import Embedder
from decoder import Decoder
from psnr import psnr
from deap import algorithms, base, creator, tools

def embed(stego, secret, chromosome):
    """Embed secret message into the host using the chromosome"""
    if len(chromosome) > 7:
        chromosome = np.packbits(chromosome.astype(np.uint8))

    # Convert to a flattened pixel sequence
    stego_sequence = MatScanner.scan_genetic(stego, chromosome)
    secret = secret.flatten()
    stego_sequence = Embedder.embed(stego_sequence, secret, chromosome)

    # Reshape the stego image
    return MatScanner.reshape_genetic(stego_sequence, stego.shape, chromosome)

def fitness(chromosome, stego, secret):
    """Computes fitness for current chromosome"""
    if len(chromosome) > 7:
        chromosome = np.packbits(chromosome.astype(np.uint8))

    # Embed the secret sequence
    try:
        stego1 = embed(stego, secret, chromosome)
    except:
        return (0,)

    return (psnr(stego, stego1),)

def decode(stego, s_shape, chromosome):
    """Decode the secret message embedded into the host image

    Args:
    	stego: stego image
    	s_shape: secret message shape
    	chromosome: solution chromosome
    
    Return:
    	np.array: the secret message
    """
    if len(chromosome) > 7:
        chromosome = np.packbits(chromosome.astype(np.uint8))

    stego = MatScanner.scan_genetic(stego, chromosome)
    secret_pixels = s_shape[0] * s_shape[1] if len(s_shape) > 1 else s_shape[0]
    secret = Decoder.decode(stego, chromosome, secret_pixels)
    return secret.reshape(s_shape)

def init_chromosome():
    c = np.array([random.randint(0, 3),
                  random.randint(0, 255),
                  random.randint(0, 255),
                  random.randint(0, 15),
                  random.randint(0, 1),
                  random.randint(0, 1),
                  random.randint(0, 1)], dtype=np.uint8)

    return creator.Individual(np.unpackbits(c))
                                                          
def cxTwoPointCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation.

    Taken from: https://github.com/DEAP/deap/blob/master/examples/ga/onemax_numpy.py#L37
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] \
        = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        
    return ind1, ind2

def main():
    #NGEN, NPOP = 200, 300
    NGEN, NPOP = 2, 10
    CXPB, MUTPB = 0.7, 0.04

    host = cv2.imread('img/Lenna-256.png', cv2.IMREAD_GRAYSCALE)
    secret = cv2.imread('img/baboon-64.png', cv2.IMREAD_GRAYSCALE)

    # Define the individuals
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Population methods
    toolbox.register('individual', init_chromosome)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register('evaluate', fitness, stego=host, secret=secret)
    toolbox.register('mate', cxTwoPointCopy)
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
    toolbox.register('select', tools.selTournament, tournsize=2)

    pop = toolbox.population(n=NPOP)

    hof = tools.HallOfFame(3, similar=np.array_equal)

    stats = tools.Statistics(lambda i : i.fitness.values)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats,
                        halloffame=hof)

    return host, secret, pop, stats, hof

if __name__ == '__main__':
    host, secret, pop, stats, hof = main()
