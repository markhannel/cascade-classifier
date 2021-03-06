Documentation
----------------

Protocol for training a classifier:

1) Use './new_cascade.sh' to generate a directory to encapsulate all information
   and images associated with this particular classifier. 
   
2) `cd` into the newly populated directory Features_\*\*\*.

3) Populate the positive and negative images directory

   Positve images should each contain exactly one instance of the feature
   you hope to detect. Negative images should have no instances of the feature
   you hope to detect. The ratio of positive to negative images will affect
   the robustness and accuracy of your classifier, however, there is no
   derivable "best" ratio.. you must experiment. There are some dubious
   sources (stackoverflow) which claim 1:2 positve to negative is a good
   ratio to shoot for.

4) Edit and run create_samples.sh

   create_samples will generate and populate a directory samples which contains 
   a set of samples generated by opencv_createsamples. Note that create_samples
   is loaded with parameters you may wish to change.

5) Edit and run train_on_samples.sh

   train_on_samples will generate a single .vec file (samples.vec) which
   will be fed into opencv_traincascade. You will have to edit this bash
   script in order to properly begin training. Namely, the number of positive
   images and negative images must be suitably changed.
