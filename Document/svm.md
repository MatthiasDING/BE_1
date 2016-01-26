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
![](https://raw.githubusercontent.com/MatthiasDING/BE_1/master/Document/dataset.jpg)

#### section 3: Split the data into train, val, and test sets
In addition we will create a small development set as a subset of the training data. we can use this for development so our code runs faster.

    imdb = split_data(imdb);

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
**The expected results**:
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

#### section 4: Preprocessing: reshape the image data into rows

    imdb.X_train = reshape(imdb.X_train, size(imdb.X_train,1), []);
    imdb.X_val   = reshape(imdb.X_val,   size(imdb.X_val  ,1), []);
    imdb.X_test  = reshape(imdb.X_test,  size(imdb.X_test ,1), []);
    imdb.X_dev   = reshape(imdb.X_dev,   size(imdb.X_dev  ,1), []);

    % As a sanity check, print out the shapes of the data
    disp('Training data shape: ');
    disp(size(imdb.X_train));
    disp('Validation data shape: ');
    disp(size(imdb.X_val));
    disp('Test data shape: ');
    disp(size(imdb.X_test));
    disp('Dev data shape: ');
    disp(size(imdb.X_dev));
    
**The expected results**:
Training data shape: 
       49000        3072

Validation data shape: 
        1000        3072

Test data shape: 
        1000        3072

Dev data shape: 
         500        3072
         
#### section 5: Preprocessing: subtract the mean image
**First**: compute the image mean based on the training data

    mean_image = mean(imdb.X_train, 1);
    disp(mean_image(1:10)); % print a few of the elements
    figure;
    imshow(uint8(reshape(mean_image, 32, 32, 3)));
    
**The expected results**: