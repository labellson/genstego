from enum import Enum
import numpy as np

class MatScanner:
    """Returns a 1-Dimensional pixel sequence giving the starting point and
    direaction.
    """

    Direction = Enum('Direction', 'raster')

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
        return np.append(mat.flatten()[idx:], mat.flatten()[:idx])

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


if __name__ == '__main__':
    direction = MatScanner.Direction.raster
    mat = np.arange(10).reshape(5,2)
    print(MatScanner.scan(mat, 4, 1, direction))
