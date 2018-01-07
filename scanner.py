from enum import Enum
import numpy as np

class MatScanner:
    """Returns a 1-Dimensional pixel sequence giving the starting point and
    direaction.
    """

    class Direction(Enum):
        raster = 0
        right_up = 1
        left_up = 2
        left_down = 3
        down_right = 4
        down_left = 5
        up_right = 6

    @staticmethod
    def _raster_scan(mat, y, x, shape):
        """Scans in raster order. From Left to Right. From Top to Down.
        
        Args:
        	mat: matrix
        	y: starting row 
        	x: starting column
        	shape: matrix shape
        
        Return:
        	np.array: flattened array from the starting point
        """
        idx = y * shape[1] + x
        return np.roll(mat, -idx).flatten()

    @staticmethod
    def _un_raster_scan(mat, y, x, shape):
        """Undo raster order"""
        idx = y * shape[1] + x
        return np.roll(mat, idx).reshape(shape)

    @staticmethod
    def _left_up(mat, y, x, shape):
        """Scans from Right to Left. From Down to Up."""
        idx = y * shape[1] + x + 1
        roll = np.roll(mat, -idx)
        return np.flip(roll.flatten(), 0)

    @staticmethod
    def _un_left_up(mat, y, x, shape):
        """Undo left up order"""
        idx = y * shape[1] + x + 1
        reshape = np.flip(mat, 0).reshape(shape)
        return np.roll(reshape, idx)

    @staticmethod
    def _left_down(mat, y, x, shape):
        """Scans from Right to Left. From Up to Down"""
        x = shape[1] - x -1
        idx = y * shape[1] + x
        flip = np.flip(mat, 1)
        return np.roll(flip, -idx).flatten()

    @staticmethod
    def _un_left_down(mat, y, x, shape):
        """Undo left down order"""
        x = shape[1] - x -1
        idx = y * shape[1] + x
        reshape = np.roll(mat.reshape(shape), idx)
        return np.flip(reshape, 1)

    @staticmethod
    def _right_up(mat, y, x, shape):
        """Scans from Left to Right. From Down to Up."""
        y = shape[0] - y - 1
        idx = y * shape[1] + x
        flip = np.flip(mat, 0)
        return np.roll(flip.flatten(), -idx)

    @staticmethod
    def _un_right_up(mat, y, x, shape):
        """Undo right up order"""
        y = shape[0] - y - 1
        idx = y * shape[1] + x
        reshape = np.roll(mat, idx).reshape(shape)
        return np.flip(reshape, 0)

    @staticmethod
    def _down_right(mat, y, x, shape):
        """Scans from Top-Down. From Left to Right"""
        idx = x * shape[0] + y
        return np.roll(mat.flatten('F'), -idx)

    @staticmethod
    def _un_down_right(mat, y, x, shape):
        """Undo down right order"""
        idx = x * shape[0] + y
        return np.roll(mat, idx).reshape(shape, order='F')

    @staticmethod
    def _down_left(mat, y, x, shape):
        """Scans from Top-Down. From Right to Left"""
        x = shape[1] - x - 1
        idx = x * shape[0] + y
        flip = np.flip(mat, 1)
        return np.roll(flip.flatten('F'), -idx)

    @staticmethod
    def _un_down_left(mat, y, x, shape):
        """Undo down left order"""
        x = shape[1] - x - 1
        idx = x * shape[0] + y
        reshape = np.roll(mat, idx).reshape(shape, order='F')
        return np.flip(reshape, 1)

    @staticmethod
    def _up_right(mat, y, x, shape):
        """Scans from Down Top. From left to right"""
        y = shape[0] - y - 1
        idx = x + shape[0] + y
        flip = np.flip(mat, 0)
        return np.roll(flip.flatten('F'), -idx)

    @staticmethod
    def _un_up_right(mat, y, x, shape):
        """Undo up right order"""
        y = shape[0] - y - 1
        idx = x + shape[0] + y
        reshape = np.roll(mat, idx).reshape(shape, order='F')
        return np.flip(reshape, 0)

    @classmethod
    def scan(cls, img, y, x, direction):
        """This method return the flattened pixel sequence given the starting
        point and direction.
       
        Args:
        	img: raw image (np.array)
        	y: starting row 
        	x: starting column
        	direction: scan direction

        Return:
        	numpy.array
        """
        direction = cls.Direction(direction)
        if direction == cls.Direction.raster:
            return cls._raster_scan(img, y, x, img.shape)
        elif direction == cls.Direction.right_up:
            return cls._right_up(img, y, x, img.shape)
        elif direction == cls.Direction.left_up:
            return cls._left_up(img, y, x, img.shape)
        elif direction == cls.Direction.left_down:
            return cls._left_down(img, y, x, img.shape)
        elif direction == cls.Direction.down_right:
            return cls._down_right(img, y, x, img.shape)
        elif direction == cls.Direction.down_left:
            return cls._down_left(img, y, x, img.shape)
        elif direction == cls.Direction.up_right:
            return cls._up_right(img, y, x, img.shape)

    @classmethod
    def scan_genetic(cls, img, chromosome):
        """This method return the flattened pixel sequence given using
        the provided chromosome

        Args:
        	img: raw image (np.array)
        	chromosome: chromosome encoding x, y, direction genes
        
        Return:
        	numpy.array
        """
        return cls.scan(img, chromosome[2], chromosome[1], chromosome[0])

    @classmethod
    def reshape(cls, img, shape, y, x, direction):
        """Returns the reshaped array given the direction.
        
        Args:
        	img: flattened image (np.array)
        	shape: Shape of array
        	y: starting row
        	x: starting column
        	direction: scan direction
        
        Return:
        	numpy.array
        """
        direction = cls.Direction(direction)
        if direction == cls.Direction.raster:
            return cls._un_raster_scan(img, y, x, shape)
        elif direction == cls.Direction.right_up:
            return cls._un_right_up(img, y, x, shape)
        elif direction == cls.Direction.left_up:
            return cls._un_left_up(img, y, x, shape)
        elif direction == cls.Direction.left_down:
            return cls._un_left_down(img, y, x, shape)
        elif direction == cls.Direction.down_right:
            return cls._un_down_right(img, y, x, shape)
        elif direction == cls.Direction.down_left:
            return cls._un_down_left(img, y, x, shape)
        elif direction == cls.Direction.up_right:
            return cls._un_up_right(img, y, x, shape)

    @classmethod
    def reshape_genetic(cls, img, shape, chromosome):
        """Returns the reshaped array using the provided chromosome
        
        Args:
        	img: flattened image (np.array)
        	shape: shape of array
        	chromosome: chromosome encoding x, y, direction genes
        
        Return:
        	numpy.array
        """
        return cls.reshape(img, shape, chromosome[2], chromosome[1], chromosome[0])

if __name__ == '__main__':
    mat = np.arange(10).reshape(2,5)
    print('- Original: \n{}'.format(mat))

    mat_scanned = MatScanner.scan(mat, 0, 3, MatScanner.Direction.raster)
    print('\n- Raster order: {}'.format(mat_scanned))
    mat_reshaped = MatScanner.reshape(mat_scanned, (2, 5), 0, 3, MatScanner.Direction.raster)
    print('- Original: \n{}'.format(mat_reshaped))

    mat_scanned = MatScanner.scan(mat, 1, 3, MatScanner.Direction.right_up)
    print('\n- Right Up order: {}'.format(mat_scanned))
    mat_reshaped = MatScanner.reshape(mat_scanned, (2, 5), 1, 3, MatScanner.Direction.right_up)
    print('- Original: \n{}'.format(mat_reshaped))

    mat_scanned = MatScanner.scan(mat, 1, 2, MatScanner.Direction.left_up)
    print('\n- Left Up order: {}'.format(mat_scanned))
    mat_reshaped = MatScanner.reshape(mat_scanned, (2, 5), 1, 2, MatScanner.Direction.left_up)
    print('- Original: \n{}'.format(mat_reshaped))

    mat_scanned = MatScanner.scan(mat, 0, 3, MatScanner.Direction.left_down)
    print('\n- Left Down order: {}'.format(mat_scanned))
    mat_reshaped = MatScanner.reshape(mat_scanned, (2, 5), 0, 3, MatScanner.Direction.left_down)
    print('- Original: \n{}'.format(mat_reshaped))

    mat_scanned = MatScanner.scan(mat, 1, 3, MatScanner.Direction.down_right)
    print('\n- Down right order: {}'.format(mat_scanned))
    mat_reshaped = MatScanner.reshape(mat_scanned, mat.shape, 1, 3, MatScanner.Direction.down_right)
    print('- Original: \n{}'.format(mat_reshaped))

    mat_scanned = MatScanner.scan(mat, 0, 3, MatScanner.Direction.down_left)
    print('\n- Down left order: {}'.format(mat_scanned))
    mat_reshaped = MatScanner.reshape(mat_scanned, mat.shape, 0, 3, MatScanner.Direction.down_left)
    print('- Original: \n{}'.format(mat_reshaped))

    mat_scanned = MatScanner.scan(mat, 0, 2, MatScanner.Direction.up_right)
    print('\n- Up Right order: {}'.format(mat_scanned))
    mat_reshaped = MatScanner.reshape(mat_scanned, mat.shape, 0, 2, MatScanner.Direction.up_right)
    print('- Original: \n{}'.format(mat_reshaped))
