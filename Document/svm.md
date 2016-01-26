# Multiclass Support Vector Machine exercise
Complete and hand in this completed worksheet. For more details see the assignments page on the course website.

In this exercise you will:

- implement a fully-vectorized **loss function** for the SVM
- implement the fully-vectorized expression for its **analytic gradient**
- **check your implementation** using numerical gradient
- use a validation set to **tune the learning rate and regularization** strength
- **optimize** the loss function with **SGD**
- **visualize** the final learned weights

#### section 1: CIFAR-10 Data Loading and Preprocessing
    imdb = load_datasets();
    
    % As a sanity check, we print out the size of the training and test data.
    disp('Training data shape: ');
    disp(size(imdb.train_data));
    disp('Training labels shape: ');
    disp(size(imdb.train_labels));
    disp('Test data shape: ');
    disp(size(imdb.test_data));
    disp('Test labels shape: ');
    disp(size(imdb.test_labels));
**The expected results**:
Training data shape: 
       50000          32          32           3

Training labels shape: 
       50000           1

Test data shape: 
       10000          32          32           3

Test labels shape: 
       10000           1
       
#### section 2: Visualize some examples from the dataset 
We show a few examples of training images from each class.

	show_datasets(imdb);
    
**The expected results**:
