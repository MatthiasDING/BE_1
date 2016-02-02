# Image features exercise
*Complete and hand in this completed worksheet (including its outputs and any supporting code outside of the worksheet) with your assignment submission. For more details see the [assignments page](http://vision.stanford.edu/teaching/cs231n/assignments.html) on the course website.*

We have seen that we can achieve reasonable performance on an image classification task by training a linear classifier on the pixels of the input image. In this exercise we will show that we can improve our classification performance by training linear classifiers not on raw pixels but on features that are computed from the raw pixels.

All of your work for this exercise will be done in this notebook.

## Load data
Similar to previous exercises, we will load CIFAR-10 data from disk.

    imdb = prepare_datasets_for_feature();
    % As a sanity check, we print out the size of the training and test data.
    disp('Training data shape: ');
    disp(size(imdb.X_train));
    disp('Training labels shape: ');
    disp(size(imdb.y_train));
    disp('Validation data shape: ');
    disp(size(imdb.X_val));
    disp('Validation labels shape: ');
    disp(size(imdb.y_val));
    disp('Test data shape: ');
    disp(size(imdb.X_test));
    disp('Test labels shape: ');
    disp(size(imdb.y_test));
    disp('==========================================');
**The expected results:**
Training data shape: 
       49000          32          32           3

Training labels shape: 
       49000           1

Validation data shape: 
        1000          32          32           3

Validation labels shape: 
        1000           1

Test data shape: 
        1000          32          32           3

Test labels shape: 
        1000           1
        
## Extract Features
For each image we will compute a Histogram of Oriented Gradients (HOG) as well as a color histogram using the hue channel in HSV color space. We form our final feature vector for each image by concatenating the HOG and color histogram feature vectors.

Roughly speaking, HOG should capture the texture of the image while ignoring color information, and the color histogram represents the color of the input image while ignoring texture. As a result, we expect that using both together ought to work better than using either alone. Verifying this assumption would be a good thing to try for the bonus section.

The `hog_feature` and `color_histogram_hsv` functions both operate on a single image and return a feature vector for that image. The extract_features function takes a set of images and a list of feature functions and evaluates each feature function on each image, storing the results in a matrix where each column is the concatenation of all feature vectors for a single image.



## Train SVM on features
Using the multiclass SVM code developed earlier in the assignment, train SVMs on top of the features extracted above; this should achieve better results than training SVMs directly on top of raw pixels.


## Bonus: Design your own features!

You have seen that simple image features can improve classification performance. So far we have tried HOG and color histograms, but other types of features may be able to achieve even better classification performance.

For bonus points, design and implement a new type of feature and use it for image classification on CIFAR-10. Explain how your feature works and why you expect it to be useful for image classification. Implement it in this notebook, cross-validate any hyperparameters, and compare its performance to the HOG + Color histogram baseline.

## Bonus: Do something extra!

Use the material and code we have presented in this assignment to do something interesting. Was there another question we should have asked? Did any cool ideas pop into your head as you were working on the assignment? This is your chance to show off!
