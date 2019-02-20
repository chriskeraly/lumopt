import numpy as np

class Wavelengths(object):
    
    def __init__(self, start, stop = None, points = 1):
        self.start = float(start)
        self.stop = float(stop) if stop else start
        if self.stop < self.start:
            raise UserWarning('span must be positive.')
        self.points = int(points)
        if self.points < 1:
            raise UserWarning('number of points must be positive.')
        if self.stop == self.start and self.points > 1:
            raise UserWarning('zero length span with multiple points.')
    
    def min(self):
        return float(self.start)

    def max(self):
        return float(self.stop)

    def __len__(self):
        return int(self.points)

    def __getitem__(self, item):
        return self.asarray()[item]

    def asarray(self):
        return np.linspace(start = self.start, stop = self.stop, num = self.points)


