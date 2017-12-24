from enum import Enum
import numpy as np

class MatScanner:
    """Returns a 1-Dimensional pixel sequence giving the starting point and
    direaction.
    """

    Direction = Enum('Direction', 'raster left_up')

    @staticmethod
    def _raster_scan(mat, y, x, shape):
        """Scans in raster order. From Left to Right. From Top to Down.
        
        Args:
        	mat: matrix
        	y: starting row 
        	x: starting column
        	shape: matrix shape
        
        Return:
        	np.array: the flattened array from the starting point
        """
        idx = y * shape[1] + x
        flatten = mat.flatten()
        return np.append(flatten[idx:], flatten[:idx])

    @staticmethod
    def _left_up(mat, y, x, shape):
        """Scans from Right to Left. From Down to Up."""
        idx = (shape[0] * shape[1]) - (y * shape[1] + x) - 1
        flip = np.flip(mat.flatten(), 0)
        return np.append(flip[idx:], flip[:idx])

    @classmethod
    def scan(cls, img, y, x, direction):
        """This method return the flattened pixel sequence given the starting
        point and direction.
       
        Args:
        	img: raw image (np.array)
        	y: starting row 
        	x: starting column
        	direction:

        Return:
        	numpy.array
        """
        if direction == cls.Direction.raster:
            return cls._raster_scan(img, y, x, img.shape)
        elif direction == cls.Direction.left_up:
            return cls._left_up(img, y, x, img.shape)


if __name__ == '__main__':
    mat = np.arange(10).reshape(5,2)
    print('- Original: {}'.format(mat))
    print('\n- Raster order: {}'.format(MatScanner.scan(mat, 4, 1, MatScanner.Direction.raster)))
    print('\n- Left Up order: {}'.format(MatScanner.scan(mat, 4, 1, MatScanner.Direction.left_up)))
