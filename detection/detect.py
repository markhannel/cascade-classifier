import cv2
import numpy as np

def bytscl(image):
    """Scales a float image to be a uint8."""
    min_  = image.min()
    max_  = image.max()
    delta = 2**8/(max_-min_)
    nuimage = (image - min_)
    nuimage *= delta
    
    return np.rint(nuimage).astype('uint8')

class Detector(object):
    """Detector loads a classifier for use in detection.

    Attributes:
        cascade: an instance of cv2.CascadeClassifier
        scale_factor: specifies how much the image size is reduced
            at each image scale. (eg 1.3 = 30% reduction). Higher
            scale factors check fewer image scales but are faster.
        min_neighbors: specifies how many neighbors each candidate
            rectangle must have for positive identification.
        min_size: the minimum possible object size.
        max_size: the maximum possible object size.
    """
    
    def __init__(self, cascade_fn='cascade_example.xml'):
        # Load cascade and check that it is functional.
        self.cascade = cv2.CascadeClassifier(cascade_fn)

        # Parameters which determine the detection behavior.
        self.scale_factor = 1.3
        self.min_neighbors = 5
        self.min_size = (10, 10)
        self.max_size = (300, 300)
        
    def detect(self, image_fn,
               scale_factor=None,
               min_neighbors=None,
               min_size=None,
               max_size=None,
               filter_type = 'std_dev',
               view_detection = False,):
        '''Returns a list rectangles (x,y,w,h) encompassing reported 
        instances of holograms in grayscale images.'''

        if scale_factor is None:
            scale_factor = self.scale_factor

        if min_neighbors is None:
            min_neighbors = self.min_neighbors
        
        if min_size is None:
            min_size = self.min_size

        if max_size is None:
            max_size = self.max_size
            
        image = cv2.imread(image_fn, 0) #'0' denotes grayscale img.
        features = self.cascade.detectMultiScale(image,
                                                 scale_factor,
                                                 min_neighbors,
                                                 minSize=min_size,
                                                 maxSize=max_size)
        
        if filter_type:
            features = self.__filter__(features, image, filter_type)
            
        if view_detection:
            self.__view__detection__(image, features)

        return features

    def __filter__(self, features, image, filter_type):
        '''Filters proposed features to hopefully reduce false positives and
        not eliminate correctly identified features.'''
        
        filters = ['std_dev']
        if filter_type not in filters:
            print('No filter type name {} present.'.format(filter_type))
            print('features returned without filtering.')
            return features

        if filter_type == 'std_dev':
            thresh = 5 # Whammy number.
            features = [(x,y,w,h) for (x,y,w,h) in features if np.std(image[y:y+h, x:x+w]) > thresh]
            return features

    def __view__detection__(self, image, features):
        '''View features.'''

        for (x,y,w,h) in features:
            cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.circle(image, (x+w/2, y+w/2), 4, (0,0,0), thickness = 2)

        cv2.imshow('Features found in image.', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def test_main():
    cascade_fn = 'cascade_example.xml'
    test_img = 'test1.png'
    detector = Detector(cascade_fn)
    detector.detect(test_img, view_detection = True)
    
if __name__ == '__main__':
    test_main()
