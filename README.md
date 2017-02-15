# cascade-classifier

<b> A module providing the ability to train cascade classifiers on holographic snapshots. </b>

## Explanation of the module.
`cascade-classifier` provides a set of bash scripts which utilize OpenCV's object detection capabilities. A cascade classifier is trained with `opencv_traincascade` over a set of positive and negative images. After proper training, the classifier can be used in Python directly with `cv2.CascadeClassifier.detectMultiScale`.

## Author List:
Aidan Abdulali - High School Student, Packer Collegiate Institute

Mark Hannel - Physics PhD Candidate, New York University

Chen Wang - Physics PhD Candidate, New York University

David Grier - Physics Professor, New York University

## Where to Get It.
[Mark Hannel's Github Repository](https://github.com/markhannel/cascade-classifier)

## Coming Soon.
1. Module for implementing object dection with a trained classifier.
2. Examples of use.
3. Profiles of detection accuracy and speed.

## Licensing.
[GPLv3](https://www.gnu.org/licenses/gpl-3.0.html)
