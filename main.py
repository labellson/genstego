import numpy as np
import random
import argparse
import helper_individual

from PIL import Image
from matplotlib import pyplot as plt
from scanner import MatScanner
from embedder import Embedder
from decoder import Decoder
from psnr import psnr
from deap import algorithms, base, creator, tools

def embed(stego, secret, chromosome):
    """Embed secret message into the host using the chromosome"""
    if len(chromosome) > 7:
        chromosome = helper_individual.packchromosome(chromosome)

    # Convert to a flattened pixel sequence
    stego_sequence = MatScanner.scan_genetic(stego, chromosome)
    secret = secret.flatten()
    stego_sequence = Embedder.embed(stego_sequence, secret, chromosome)

    # Reshape the stego image
    return MatScanner.reshape_genetic(stego_sequence, stego.shape, chromosome)

def fitness(chromosome, stego, secret):
    """Computes fitness for current chromosome"""
    if len(chromosome) > 7:
        chromosome = helper_individual.packchromosome(chromosome)

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
        chromosome = helper_individual.packchromosome(chromosome)

    stego = MatScanner.scan_genetic(stego, chromosome)
    secret_pixels = s_shape[0] * s_shape[1] if len(s_shape) > 1 else s_shape[0]
    secret = Decoder.decode(stego, chromosome, secret_pixels)
    return secret.reshape(s_shape)


def imshow(host, stego, secret):
    """Show the images with matplotlib"""
    fig, axes = plt.subplots(1,3)

    axes[0].set_title('Host')
    axes[1].set_title('Stego')
    axes[2].set_title('Secret')

    axes[0].imshow(host, cmap='gray', aspect='equal')
    axes[1].imshow(stego, cmap='gray', aspect='equal')
    axes[2].imshow(secret, cmap='gray', aspect='equal')

    plt.setp(axes, xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()

def init_chromosome():
    return creator.Individual(helper_individual.init_chromosome())

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

def setup_deap_individuals():
    # Define the individuals
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', np.ndarray, fitness=creator.FitnessMax)

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument('-ht', '--host', required=True)
    ap.add_argument('-s', '--secret', required=True)
    ap.add_argument('-g', '--generations', default=80, type=int)
    ap.add_argument('-p', '--population', default=100, type=int)
    ap.add_argument('-c', '--crossover', default=0.7, type=int)
    ap.add_argument('-m', '--mutation', default=0.25, type=int)

    args = vars(ap.parse_args())

    NGEN, NPOP, LAMBDA = args['generations'], args['population'], 100
    CXPB, MUTPB = args['crossover'], args['mutation']
    ICXPB, IMUTPB = 0.5, 0.2

    # Convert to grayscale: http://pillow.readthedocs.io/en/5.0.0/handbook/concepts.html#concept-modes
    host = np.array(Image.open(args['host']).convert('L'))
    secret = np.array(Image.open(args['secret']).convert('L'))

    setup_deap_individuals()

    toolbox = base.Toolbox()

    # Population methods
    toolbox.register('individual', init_chromosome)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register('evaluate', fitness, stego=host, secret=secret)
    toolbox.register('mate', cxTwoPointCopy)
    toolbox.register('mutate', tools.mutFlipBit, indpb=IMUTPB)
    toolbox.register('select', tools.selTournament, tournsize=2)

    pop = toolbox.population(n=NPOP)

    hof = tools.HallOfFame(3, similar=np.array_equal)

    stats = tools.Statistics(lambda i : i.fitness.values)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=hof)

    # Embed secret image using the best individual
    stego = embed(host, secret, hof.items[0])

    # Show the best solution
    imshow(host, stego, secret)

    return host, stego, secret, pop, stats, logbook, hof

if __name__ == '__main__':
    host, stego, secret, pop, stats, logbook, hof = main()
    attrs = {
        'host' : host,
        'stego' : stego,
        'secret' : secret,
        'pop' : pop,
        'logbook' : logbook,
        'hof' : hof
    }
