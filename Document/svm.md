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
  130.6419  130.0241  129.6634  129.4197  129.1715  128.8883  128.4675  127.9940  127.4662  127.2436
![](https://raw.githubusercontent.com/MatthiasDING/BE_1/master/Document/svm_mean_image.jpg)

**Second**: subtract the mean image from train and test data

    imdb.X_train = bsxfun(@minus, imdb.X_train, mean_image);
    imdb.X_val   = bsxfun(@minus, imdb.X_val  , mean_image);
    imdb.X_test  = bsxfun(@minus, imdb.X_test , mean_image);
    imdb.X_dev   = bsxfun(@minus, imdb.X_dev  , mean_image);
    
**Third**: append the bias dimension of ones (i.e. bias trick) so that our SVM only has to worry about optimizing a single weight matrix W.

    imdb.X_train = cat(2, imdb.X_train, ones(size(imdb.X_train, 1), 1));
    imdb.X_val   = cat(2, imdb.X_val  , ones(size(imdb.X_val  , 1), 1));
    imdb.X_test  = cat(2, imdb.X_test , ones(size(imdb.X_test , 1), 1));
    imdb.X_val   = cat(2, imdb.X_val  , ones(size(imdb.X_val  , 1), 1));

##SVM Classifier
Your code for this section will all be written inside the folder **./classifier/svm**.
**Step 1**: As you can see, we have prefilled the function ***svm_loss_naive*** which uses for loops to evaluate the multiclass SVM loss function. 

    %Evaluate the naive implementation of the loss we provided for you:
    W = randn(10, 3073) * 0.0001;
    [loss, dW] = svm_loss_naive(W, imdb.X_train, imdb.y_train, 0.00001);

    fprintf('loss: %f\n',loss);
**The result (random W) looks like**:
loss: 9.390128
    
The grad returned from the function above is right now all zero. Derive and implement the gradient for the SVM cost function and implement it inline inside the function ***svm_loss_naive***. You will find it helpful to interleave your new code inside the existing function.

To check that you have correctly implemented the gradient correctly, you can numerically estimate the gradient of the loss function and compare the numeric estimate to the gradient that you computed. We have provided code that does this for you:

Once you've implemented the gradient, recompute it with the code below and gradient check it with the function we provided for you

    % Compute the loss and its gradient at W.
    [loss, grad] = svm_loss_naive(W, imdb.X_train, imdb.y_train, 0.0);

Numerically compute the gradient along several randomly chosen dimensions, and compare them with your analytically computed gradient. The numbers should match almost exactly along all dimensions.

    f = @(x)svm_loss_naive(x, imdb.X_train, imdb.y_train, 0.0);
    grad_check_sparse(f, W, grad, 10);
**The result (random W) looks like**:
numerical: 15.958773 analytic: 15.956062, relative error: 8.492935e-05
numerical: 4.287074 analytic: 4.288790, relative error: 2.000480e-04
numerical: -4.972506 analytic: -4.974457, relative error: 1.961668e-04
numerical: 1.652603 analytic: 1.654120, relative error: 4.587633e-04
numerical: -18.316459 analytic: -18.312615, relative error: 1.049457e-04
numerical: 11.347319 analytic: 11.351546, relative error: 1.862250e-04
numerical: 5.228363 analytic: 5.228370, relative error: 7.434179e-07
numerical: -28.642817 analytic: -28.646421, relative error: 6.291637e-05
numerical: -1.875807 analytic: -1.874700, relative error: 2.950681e-04
numerical: 14.792061 analytic: 14.793707, relative error: 5.563963e-05


**Question 1**: It is possible that once in a while a dimension in the gradcheck will not match exactly. What could such a discrepancy be caused by? Is it a reason for concern? What is a simple example in one dimension where a gradient check could fail? Hint: the SVM loss function is not strictly speaking differentiable
**Your answer**:

**Step 2**: Next implement the function ***svm_loss_vectorized***. For now only compute the loss, we will implement the gradient in a moment.

    tic;
    [loss_naive, ~] = svm_loss_naive(W, imdb.X_train, imdb.y_train, 0.00001);
    time = toc;
    fprintf('Naive loss: %e computed in %fs\n', loss_naive, time);

    tic;
    [loss_vectorized, ~] = svm_loss_vectorized(W, imdb.X_train, imdb.y_train, 0.00001);
    time = toc;
    fprintf('Vectorized loss: %e computed in %fs\n', loss_vectorized, time);

    %The losses should match but your vectorized implementation should be much faster.
    fprintf('difference: %f\n', loss_naive - loss_vectorized);
    
**The result (random W) looks like**:
Naive loss: 9.390128e+00 computed in 40.616371s
Vectorized loss: 9.390128e+00 computed in 0.300807s
difference: 0.000000

    
Complete the implementation of ***svm_loss_vectorized***, and compute the gradient of the loss function in a vectorized way.
	
    % The naive implementation and the vectorized implementation should match, but
    % the vectorized version should still be much faster.
    tic;
    [~, grad_naive] = svm_loss_naive(W, imdb.X_train, imdb.y_train, 0.00001);
    time = toc;
    fprintf('Naive loss and gradient: computed in %fs\n',time);

    tic;
    [~, grad_vectorized] = svm_loss_vectorized(W, imdb.X_train, imdb.y_train, 0.00001);
    time = toc;
    fprintf('Vectorized loss and gradient: computed in %fs\n', time);

    % The loss is a single number, so it is easy to compare the values computed
    % by the two implementations. The gradient on the other hand is a matrix, so
    % we use the Frobenius norm to compare them. 
    difference = norm(grad_naive - grad_vectorized, 'fro');
    fprintf('difference: %f\n', difference);
**The result (random W) looks like**:
Naive loss and gradient: computed in 40.651275s
Vectorized loss and gradient: computed in 0.305512s
difference: 0.000000

## Stochastic Gradient Descent
We now have vectorized and efficient expressions for the loss, the gradient and our gradient matches the numerical gradient. We are therefore ready to do SGD to minimize the loss.

Now implement SGD in ***linear_svm_train()*** function and run it with the code below.

    tic;
    [model, hist] = linear_svm_train(imdb.X_train, imdb.y_train, 1e-7, 5e4, 1500, 200, 1);
    time = toc;
    fprintf('That took %fs\n', time);

**The result looks like**:   
