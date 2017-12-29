import numpy as np

class Decoder:
    """Methods to decode secret messages from a host image"""

    @staticmethod
    def _decode(stego, secret, idx):
        return np.append(secret, stego[idx])
    
    @classmethod
    def decode(cls, stego, chromosome, npixel):
        """Embed secret bits into stego bits according to the mask
        The chromosome has the following gene representation:
        [dir, xoffset, yoffset, bit-planes, sb-pole, sb-dire, bp-dire]

        Args:
        	stego: Stego pixel sequence 
        	chromosome: Chromosome of the GA
        
        Return:
        	numpy.array: secret bit sequence
        """
        # Bit-Planes: Extract the bit mask
        mask = np.unpackbits(np.array([chromosome[3]], dtype='uint8'))[4:]
        idx = np.argwhere(mask == True)
        capacity = round(8 / len(idx)) * npixel

        # BP-Dire: Use LSB or MSB
        if chromosome[6]:
            idx += 4

        # Convert data to uint8
        stego = stego.astype('uint8')

        # Stego bits [nbits]
        stego = np.unpackbits(stego)

        # Create the iterator for the stereo image
        it = np.nditer(stego, flags=['external_loop', 'buffered'],
                       op_flags=['readwrite'], buffersize=8)
        # Decode
        secret = np.array([], dtype=np.uint8)
        for _ in range(capacity):
            secret = cls._decode(it.value, secret, idx)
            it.iternext()

        # Check if secret is mod 8
        mod = len(secret) % 8
        if not mod == 0:
            secret = np.delete(secret, np.s_[-mod:])
        
        secret = np.packbits(secret)

        # SB-Pole: Compliment secret bits
        if chromosome[4]:
            np.invert(secret, secret)

        # SB-Dire: reverse the secret sequence
        if chromosome[5]:
            secret = secret[::-1]

        return secret
