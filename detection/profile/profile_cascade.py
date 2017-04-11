import numpy as np
import os
import sys
import cv2 
import time
import matplotlib.pyplot as plt


def profile(cascadename, test_img_dir, result_name):
    '''Profile runs a cascade over the images in test_img_dir and
    generates a file with the time spent on each frame and the 
    center of each feature.'''

    # Launch cascade 
    cascade = cv2.CascadeClassifier(cascadename)

    # Gather images
    images= [ img for img in os.listdir(test_img_dir) if img.endswith('.png')]

    images = np.sort(images)

    experiment=[]
    # For each image, perform feature identification
    for image in images:
        print image
        frame=[]

        # Make filename
        filename = image.replace('.png', '')
    
        # Read in image
        img = cv2.imread(test_img_dir + image,0)
    
        # Perform cascade
        start_time = time.time()
        features = cascade.detectMultiScale(img, 1.3, 5)
        time1= (time.time() - start_time)

        # Draw Rectangle
        start_time = time.time()
        actual_feature = [False for i in xrange(len(features))]
        for i, (x,y,w,h) in enumerate(features):
            std_test = np.std(img[y:y+h, x:x+w])
            std_test = int(std_test)
            if std_test > 13:
                actual_feature[i] = True
        time2= ((time.time() - start_time))
    
        # Save time spent on computation.
        frame.append(time1+time2)
    
        for i, (x,y,w,h) in enumerate(features):
            if actual_feature[i]:
                frame.append(x+w/2)
                frame.append(y+h/2)
        experiment.append(frame) # Save frame results.
    
    # Write results to a file.
    with open(result_name, 'w') as f:
        # Example of line: [0.026, 480, 215, 230, 615]
        for aframe in experiment:
            entry = ' '.join(map(str, aframe))
            f.write(entry)
            f.write('\n') # Every frame's data should have it's own line

if __name__ == '__main__':
    # Begin argparse.
    import argparse
    parser = argparse.ArgumentParser()

    # Prepare for default arguments.
    root = '/home/mark/Github/cascade-classifier/detection/'
    cascadename  = root + 'cascade_example.xml'
    test_img_dir = root + 'test_imgs/diffusion/'
    result_name  = test_img_dir + 'fit_result.dat'

    # Initialize parser args.
    parser.add_argument('-cascadename', type=str, help='Name of the cascade',
                        default=cascadename)
    parser.add_argument('-test_img_dir', type=str, help='Directory with the test data',
                        default=test_img_dir)
    parser.add_argument('-result_name', type=str, help='Name of the cascade',
                        default=result_name)
    args = parser.parse_args()

    # Test that result name isn't already used.
    if os.path.isfile(args.result_name):
        print '**Warning**'
        print 'Request resultname already exists. Choose a new name or displace the original file.'
        print ' '
        raise NameError

    # Profile the code over args arguments.
    profile(args.cascadename, args.test_img_dir, args.result_name)
