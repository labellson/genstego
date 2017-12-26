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
        pass

    @classmethod
    def embed(cls, stego, secret, mask, lsb=True):
        """Embed secret bits into stego bits according to the mask

        Args:
        	stego: Stego pixel sequence 
        	secret: Secret pixel sequence
        	mask: bits to use
        
        Return:
        	numpy.array: stego bit sequence with embedded bits
        """
        idx = np.argwhere(mask == True)
        capacity = round(8 / len(idx)) * len(secret)

        if capacity > stego.shape[0]:
            raise cls.EmbeddingError('Insufficient stego pixel size.')

        # Secret bitarray [nbits]
        secret_bits = np.unpackbits(secret.astype('uint8'))

        # Stego bits [nbits]
        stego_bits = np.unpackbits(stego.astype('uint8'))

        # Embed the secret bits into the stego image
        it = np.nditer(stego_bits, flags=['external_loop', 'buffered'],
                       op_flags=['readwrite'], buffersize=8)

        while not it.finished:
            cls._embed(it.iternext().value[...], secret_bits, idx)
