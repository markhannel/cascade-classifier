#!/bin/bash

echo "Preparing for training..."

echo "Merging samples into a single vector."
python ../bin/mergevec.py -v samples/ -o samples.vec

echo "Beginning Training..."
mkdir classifier
opencv_traincascade -data classifier \
		    -vec samples.vec \
		    -bg negatives.dat \
		    -numStages 10 \
		    -minHitRate 0.999 \
		    -maxFalseAlarmRate 0.5 \
		    -numPos 600 \
		    -numNeg 3000 \
		    -w 90 -h 90 \
		    -mode ALL \
		    -precalcValBufSize 2048 \
		    -precalcIdxBufSize 2048
