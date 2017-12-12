from enum import Enum
import numpy as np

class Scanner():
    """Acts like an iterator for the stego images.
    
    scan method returns a pixel sequence given the starting point and 
    direction.
    """

    Direction = Enum('Direction', 'raster')
    
    def __init__(self, img, start_point, direction):
        """Args:
        	start_point [x, y]: starting pixel of the scanner
        	direction: scan direction
        """
        self._start_point = start_point
        self._started = False

        self.finished = False
        self.img = img
        self.current = start_point
        self.direction = direction

    def scan(self):
        """This method return the pixel index sequence given the starting
        point and direction.
       
        Args:
        	start_point (x, y): starting pixel of the scanner
        	direction:

        Return:
        	numpy.array
        """
        if self.finished:
            return None

        buff = self._raster_scan(self.img.shape)

        if not self._started:
            self._started = True

        return buff

    def _raster_scan(self, s):
        """Scans in raster order. From Left to Right. From Top to Down.
        
        Args:
        	s: shape of img
        
        Return:
        	np.array: the row in case that current column is 0, or the remaining
        		row if not.
        """
        # Finish condition for the iterator
        if self._started and self.current[0] == self._start_point[0]:
            self.finished = True
            return self.img[self.current[0],
                            self.current[1] : self._start_point[1]]
            
        # Normal raster scan
        row = self.img[self.current[0], self.current[1]:]

        # Increment the current index on raster scan
        if self.current[0] == s[0] - 1:
            self.current = [0, 0]
        else:
            self.current = [self.current[0] + 1, 0]

        # Another finish condition for the iterator
        if self._started and self.current == self._start_point:
            self.finished = True

        return row
