import numpy as np

class Embedder:

    class EmbeddingError(Exception):
        """Exception raised for embedding errors"""
        
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return repr(self.value)

    @staticmethod
    def _embed(stego, secret, idx):
        """Embed the secret bits in stego[idx]"""
        np.put(stego, idx, secret)

    @classmethod
    def embed(cls, stego, secret, chromosome):
        """Embed secret bits into stego bits according to the mask
        The chromosome has the following gene representation:
        [dir, xoffset, yoffset, bit-planes, sb-pole, sb-dire, bp-dire]

        Args:
        	stego: Stego pixel sequence 
        	secret: Secret pixel sequence
        	chromosome: Chromosome of the GA
        
        Return:
        	numpy.array: stego bit sequence with embedded bits
        """
        # Bit-Planes: Extract the bit mask
        mask = np.unpackbits(np.array([chromosome[3]], dtype='uint8'))[4:]
        idx = np.argwhere(mask == True)
        capacity = round(8 / len(idx)) * len(secret)

        if capacity > stego.shape[0]:
            raise cls.EmbeddingError('Insufficient stego pixel size.')

        # Convert data to uint8
        stego = stego.astype('uint8')
        secret = secret.astype('uint8')

        # SB-Pole: Compliment secret bits
        if chromosome[4]:
            np.invert(secret, secret)

        # SB-Dire: reverse the secret sequence
        if chromosome[5]:
            secret = secret[::-1]

        # BP-Dire: Use LSB or MSB
        if chromosome[6]:
            idx += 4

        # Secret bitarray [nbits]
        secret = np.unpackbits(secret)

        # Stego bits [nbits]
        stego = np.unpackbits(stego)

        # Embed the secret bits into the stego image
        it = np.nditer(stego, flags=['external_loop', 'buffered'],
                       op_flags=['readwrite'], buffersize=8)

        while len(secret) > 0:
            cls._embed(it.value[...], secret[:len(idx)], idx)
            secret = np.delete(secret, np.s_[:len(idx)])
            it.iternext()

        return np.packbits(stego)
