import numpy as np
import random

# Chromosome representation
c_rep = [4, 8, 8, 4, 1, 1, 1]

def init_gen(length):
    g = list()
    for _ in range(length):
        g.append(random.randint(0, 1))

    return g

def init_chromosome():
    c = list()
    c.extend(init_gen(4)) # direction 4 bits
    c.extend(init_gen(8)) # x-offset 8 bits
    c.extend(init_gen(8)) # y-offset 8 bits
    c.extend(init_gen(4)) # bit-planes 4 bits
    c.extend(init_gen(1)) # sb-pole 1 bit
    c.extend(init_gen(1)) # sb-dire 1 bit
    c.extend(init_gen(1)) # bp-dire 1 bit

    return np.array(c , dtype=np.uint8)

def packchromosome(chromosome):
    """Convert the base 2 chromosome to base 10"""
    _chromosome = np.zeros((len(c_rep), 8), dtype=np.uint8)

    j = 0
    for i, c in zip(c_rep, _chromosome):
        c[-i:] = chromosome[j : j + i]
        j += i

    return np.packbits(_chromosome)

def unpackchromosome(chromosome):
    """Convert the base 10 chromosome to base 2"""
    chromosome = np.unpackbits(chromosome.reshape(-1, 1), 1)
    _chromosome = np.array(list(), dtype=np.uint8)
    
    for i, c in zip(c_rep, chromosome):
        _chromosome = np.append(_chromosome, c[-i:])

    return _chromosome
