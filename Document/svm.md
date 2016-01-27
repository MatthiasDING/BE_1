# Multiclass Support Vector Machine exercise
Complete and hand in the script ***Run_svm.m*** and other functions in the folder ***./classifier/svm***. For more details see the assignments page on the course website.

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
    imdb.X_dev   = cat(2, imdb.X_dev  , ones(size(imdb.X_dev  , 1), 1));

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


#### Question 1: 
It is possible that once in a while a dimension in the gradcheck will not match exactly. What could such a discrepancy be caused by? Is it a reason for concern? What is a simple example in one dimension where a gradient check could fail? Hint: the SVM loss function is not strictly speaking differentiable
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

**Step 1:** Now implement SGD in ***linear_svm_train()*** function and run it with the code below.

    tic;
    [model, hist] = linear_svm_train(imdb.X_train, imdb.y_train, 1e-7, 5e4, 1500, 200, 1);
    time = toc;
    fprintf('That took %fs\n', time);

**The result looks like**:   
iteration 100 / 1500: loss 289.964267
iteration 200 / 1500: loss 109.307092
iteration 300 / 1500: loss 42.708123
iteration 400 / 1500: loss 18.763980
iteration 500 / 1500: loss 10.996310
iteration 600 / 1500: loss 7.378142
iteration 700 / 1500: loss 6.038593
iteration 800 / 1500: loss 5.480713
iteration 900 / 1500: loss 5.292395
iteration 1000 / 1500: loss 5.740316
iteration 1100 / 1500: loss 5.282647
iteration 1200 / 1500: loss 4.878409
iteration 1300 / 1500: loss 5.803949
iteration 1400 / 1500: loss 5.732068
iteration 1500 / 1500: loss 5.790276
That took 11.883061s

A useful debugging strategy is to plot the loss as a function of iteration number:

    figure;
    plot(loss_hist);
    xlabel('Iteration number');
    ylabel('Loss Value');
    
**The result looks like**:   
![](https://raw.githubusercontent.com/MatthiasDING/BE_1/master/Document/svm_loss_hist.jpg)

Write the linear_svm_predict function and evaluate the performance on both the training and validation set

    y_train_pred = linear_svm_predict(model, imdb.X_train);
    fprintf('training accuracy: %f\n', mean(imdb.y_train == y_train_pred'));
    y_val_pred = linear_svm_predict(model, imdb.X_val);
    fprintf('validation accuracy: %f\n', mean(imdb.y_val == y_val_pred'));
**The result looks like**: 
training accuracy: 0.371265
validation accuracy: 0.376000

**Step 2:** Use the validation set to tune hyperparameters (regularization strength and learning rate). You should experiment with different ranges for the learning rates and regularization strengths; if you are careful you should be able to get a classification accuracy of about 0.4 on the validation set.

    learning_rates = [1e-7, 2e-7, 3e-7, 5e-5, 8e-7];
    regularization_strengths = [1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 1e5];
Results is dictionary mapping tuples of the form (learning_rate, regularization_strength) to tuples of the form (training_accuracy, validation_accuracy). The accuracy is simply the fraction of data points that are correctly classified.

    results = zeros(length(learning_rates), length(regularization_strengths), 2);
    best_val = -1;   %The highest validation accuracy that we have seen so far.
    best_svm = struct(); %The LinearSVM model that achieved the highest validation rate.

**Todo: now implement *Run_svm.m* to find the best hyperparameters**

    % ################################################################################
    % # Write code that chooses the best hyperparameters by tuning on the validation #
    % # set. For each combination of hyperparameters, train a linear SVM on the      #
    % # training set, compute its accuracy on the training and validation sets, and  #
    % # store these numbers in the results dictionary. In addition, store the best   #
    % # validation accuracy in best_val and the LinearSVM model that achieves this  #
    % # accuracy in best_svm.                                                        #
    % #                                                                              #
    % # Hint: You should use a small value for num_iters as you develop your         #
    % # validation code so that the SVMs don't take much time to train; once you are #
    % # confident that your validation code works, you should rerun the validation   #
    % # code with a larger value for num_iters.                                      #
    % ################################################################################
    
    	Your code 
        
	% ################################################################################
    % #                              END OF YOUR CODE                                #
    % ################################################################################
    
**Print out results :**

    for i =1:length(learning_rates)
        for j= 1:length(regularization_strengths)
             fprintf('lr %e reg %e train accuracy: %f val accuracy: %f\n', ...
                 learning_rates(i), regularization_strengths(j), results(i,j,1), results(i,j,2));
        end
    end
    fprintf('best validation accuracy achieved during cross-validation: %f\n', best_val);
**The result looks like**:
lr 1.000000e-07 reg 1.000000e+04 train accuracy: 0.348122 val accuracy: 0.342000
lr 1.000000e-07 reg 2.000000e+04 train accuracy: 0.340898 val accuracy: 0.342000
lr 1.000000e-07 reg 3.000000e+04 train accuracy: 0.345163 val accuracy: 0.339000
lr 1.000000e-07 reg 4.000000e+04 train accuracy: 0.341673 val accuracy: 0.361000
lr 1.000000e-07 reg 5.000000e+04 train accuracy: 0.344000 val accuracy: 0.346000
lr 1.000000e-07 reg 6.000000e+04 train accuracy: 0.344102 val accuracy: 0.351000
lr 1.000000e-07 reg 7.000000e+04 train accuracy: 0.339551 val accuracy: 0.336000
lr 1.000000e-07 reg 8.000000e+04 train accuracy: 0.341571 val accuracy: 0.360000
lr 1.000000e-07 reg 1.000000e+05 train accuracy: 0.340143 val accuracy: 0.353000
lr 2.000000e-07 reg 1.000000e+04 train accuracy: 0.381959 val accuracy: 0.400000
lr 2.000000e-07 reg 2.000000e+04 train accuracy: 0.384020 val accuracy: 0.376000
lr 2.000000e-07 reg 3.000000e+04 train accuracy: 0.375571 val accuracy: 0.386000
lr 2.000000e-07 reg 4.000000e+04 train accuracy: 0.370857 val accuracy: 0.382000
lr 2.000000e-07 reg 5.000000e+04 train accuracy: 0.372224 val accuracy: 0.376000
lr 2.000000e-07 reg 6.000000e+04 train accuracy: 0.375714 val accuracy: 0.408000
lr 2.000000e-07 reg 7.000000e+04 train accuracy: 0.374714 val accuracy: 0.383000
lr 2.000000e-07 reg 8.000000e+04 train accuracy: 0.381571 val accuracy: 0.397000
lr 2.000000e-07 reg 1.000000e+05 train accuracy: 0.379000 val accuracy: 0.376000
lr 3.000000e-07 reg 1.000000e+04 train accuracy: 0.361449 val accuracy: 0.375000
lr 3.000000e-07 reg 2.000000e+04 train accuracy: 0.352612 val accuracy: 0.373000
lr 3.000000e-07 reg 3.000000e+04 train accuracy: 0.356612 val accuracy: 0.363000
lr 3.000000e-07 reg 4.000000e+04 train accuracy: 0.371714 val accuracy: 0.363000
lr 3.000000e-07 reg 5.000000e+04 train accuracy: 0.365592 val accuracy: 0.356000
lr 3.000000e-07 reg 6.000000e+04 train accuracy: 0.366367 val accuracy: 0.353000
lr 3.000000e-07 reg 7.000000e+04 train accuracy: 0.352531 val accuracy: 0.348000
lr 3.000000e-07 reg 8.000000e+04 train accuracy: 0.360143 val accuracy: 0.378000
lr 3.000000e-07 reg 1.000000e+05 train accuracy: 0.360041 val accuracy: 0.359000
lr 5.000000e-05 reg 1.000000e+04 train accuracy: 0.067816 val accuracy: 0.088000
lr 5.000000e-05 reg 2.000000e+04 train accuracy: 0.077122 val accuracy: 0.078000
lr 5.000000e-05 reg 3.000000e+04 train accuracy: 0.072939 val accuracy: 0.069000
lr 5.000000e-05 reg 4.000000e+04 train accuracy: 0.049367 val accuracy: 0.048000
lr 5.000000e-05 reg 5.000000e+04 train accuracy: 0.050245 val accuracy: 0.045000
lr 5.000000e-05 reg 6.000000e+04 train accuracy: 0.100980 val accuracy: 0.105000
lr 5.000000e-05 reg 7.000000e+04 train accuracy: 0.120878 val accuracy: 0.124000
lr 5.000000e-05 reg 8.000000e+04 train accuracy: 0.052204 val accuracy: 0.047000
lr 5.000000e-05 reg 1.000000e+05 train accuracy: 0.066714 val accuracy: 0.052000
lr 8.000000e-07 reg 1.000000e+04 train accuracy: 0.285490 val accuracy: 0.278000
lr 8.000000e-07 reg 2.000000e+04 train accuracy: 0.304000 val accuracy: 0.314000
lr 8.000000e-07 reg 3.000000e+04 train accuracy: 0.296286 val accuracy: 0.295000
lr 8.000000e-07 reg 4.000000e+04 train accuracy: 0.320184 val accuracy: 0.347000
lr 8.000000e-07 reg 5.000000e+04 train accuracy: 0.307612 val accuracy: 0.322000
lr 8.000000e-07 reg 6.000000e+04 train accuracy: 0.320510 val accuracy: 0.306000
lr 8.000000e-07 reg 7.000000e+04 train accuracy: 0.314388 val accuracy: 0.314000
lr 8.000000e-07 reg 8.000000e+04 train accuracy: 0.319143 val accuracy: 0.302000
lr 8.000000e-07 reg 1.000000e+05 train accuracy: 0.296122 val accuracy: 0.297000
best validation accuracy achieved during cross-validation: 0.408000

**Visualize the cross-validation results**

    % plot training accuracy
    [x_scatter, y_scatter] = meshgrid(log(learning_rates), log(regularization_strengths));
    x_scatter = reshape(x_scatter, 1, []);
    y_scatter = reshape(y_scatter, 1, []);

    figure;
    hold on;
    marker_size = 100;
    colors = reshape(results(:,:,1), 1, []);
    subplot(2, 1, 1)
    scatter(x_scatter, y_scatter, marker_size, colors, 'filled');
    colorbar();
    xlabel('log learning rate');
    ylabel('log regularization strength');
    title('CIFAR-10 training accuracy');

    %plot validation accuracy
    colors = reshape(results(:,:,2), 1, []);
    subplot(2, 1, 2)
    scatter(x_scatter, y_scatter, marker_size, colors, 'filled');
    colorbar();
    xlabel('log learning rate');
    ylabel('log regularization strength');
    title('CIFAR-10 validation accuracy')
    hold off;
    
**Evaluate the best svm on test set**

    y_test_pred = linear_svm_predict(best_svm, imdb.X_test);
    test_accuracy = mean(imdb.y_test == y_test_pred');
    fprintf('linear SVM on raw pixels final test set accuracy: %f\n', test_accuracy);
    
**The result looks like**:
linear SVM on raw pixels final test set accuracy: 0.359000

**Visualize the learned weights for each class.**

    %Depending on your choice of learning rate and regularization strength, these may
    %or may not be nice to look at.
    w = best_svm.W(:,1:end-1); % strip out the bias
    w = reshape(w,10, 32, 32, 3);
    w_min = min(w(:));
    w_max = max(w(:));
    classes = imdb.class_names;

    figure;
    hold on;
    for i = 1:10
      subplot(2, 5, i)    
      % Rescale the weights to be between 0 and 255
      wimg = 255.0 * (squeeze(w(i,:,:,:)) - w_min) / (w_max - w_min);
      imshow(uint8(wimg));
      axis('off');
      title(classes{i});
    end

**The result looks like**:

####Question 2:
Describe what your visualized SVM weights look like, and offer a brief explanation for why they look they way that they do.
**Your answer:** 