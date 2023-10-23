import cv2
import numpy as np
class BGR2RGB565(object):

    def __init__(self):
        pass

    def __call__(self, sample):
        numpy_array = np.asarray(sample)
        image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR565) # convert BGR to RGB565
        return image