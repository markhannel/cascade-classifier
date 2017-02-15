#!/bin/bash

echo "Preparing to generate samples..."

echo "Collecting negative and positive image names to be processed"
find positive_images/ -iname "*.jpg" > positives.dat
find negative_images/ -iname "*.png" > negatives.dat


# Generate samples with noted parameters
mkdir samples
perl ../bin/createsamples.pl positives.dat negatives.dat samples 1500 \
"opencv_createsamples -bgcolor 0 -maxxangle 0.1 -maxyangle 0.1 \
-maxzangle 0.1 -maxidev 40 -w 90 -h 90"
