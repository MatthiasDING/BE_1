# Softmax exercise

Complete and hand in the script **Run_softmax.m** and other functions in the folder **"./classifier/softmax"**. For more details see the [assignments page](http://vision.stanford.edu/teaching/cs231n/assignments.html) on the course website.

This exercise is analogous to the SoftMax exercise. You will:

- implement a fully-vectorized **loss function** for the Softmax classifier
- implement the fully-vectorized expression for its **analytic gradient**
- **check your implementation** with numerical gradient
- use a validation set to **tune the learning rate and regularization** strength
- **optimize** the loss function with **SGD**
- **visualize** the final learned weights

##Prepare datasets
    %% Load and preprocess the raw CIFAR-10 data.
    imdb = prepare_datasets();

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
    disp('==========================================')
**The expected results**:
Training data shape: 
       49000        3073

Training labels shape: 
       49000           1

Validation data shape: 
        1000        3073

Validation labels shape: 
        1000           1

Test data shape: 
        1000        3073

Test labels shape: 
        1000           1


## Softmax Classifier

Your code for this section will all be written inside the folder **./classifier/softmax**.
First implement the naive softmax loss function with nested loops. Open the file **./classifiers/softmax/softmax_loss_naive.m** and implement the **softmax_loss_naive** function.

    % Generate a random softmax weight matrix and use it to compute the loss.
    W = randn(10, 3073) * 0.0001;
    [loss, grad] = softmax_loss_naive(W, imdb.X_train, imdb.y_train, 0.0);

    % As a rough sanity check, our loss should be something close to -log(0.1).
    fprintf('loss: %f\n', loss);
    fprintf('sanity check: %f\n', (-log(0.1)));
**The result (random W) looks like**:
loss: 2.368984
sanity check: 2.302585

#### Inline Question 1:
Why do we expect our loss to be close to -log(0.1)? Explain briefly.
**Your answer:**

Complete the implementation of **softmax_loss_naive** and implement a (naive) version of the gradient that uses nested loops.

	[loss, grad] = softmax_loss_naive(W, imdb.X_train, imdb.y_train, 0.0);
    
As we did for the SVM, use numeric gradient checking as a debugging tool. The numeric gradient should be close to the analytic gradient.

    f = @(x)softmax_loss_naive(x, imdb.X_train, imdb.y_train, 0.0);
    grad_check_sparse(f, W, grad, 10);
    
**The result (random W) looks like**:
numerical: 0.909104 analytic: 0.909103, relative error: 4.420315e-08
numerical: -0.456601 analytic: -0.456601, relative error: 1.347291e-08
numerical: -1.821526 analytic: -1.821526, relative error: 1.782215e-08
numerical: -1.209214 analytic: -1.209214, relative error: 3.648012e-08
numerical: 0.945788 analytic: 0.945788, relative error: 7.806608e-09
numerical: -1.337273 analytic: -1.337273, relative error: 7.051367e-09
numerical: 0.161232 analytic: 0.161232, relative error: 7.258902e-08
numerical: -1.057154 analytic: -1.057154, relative error: 4.726788e-08
numerical: -1.268697 analytic: -1.268697, relative error: 8.034296e-09
numerical: 0.231519 analytic: 0.231519, relative error: 2.203893e-07

Similar to SVM case, do another gradient check with regularization

    f = @(x)softmax_loss_naive(x, imdb.X_train, imdb.y_train, 1e2);
    grad_check_sparse(f, W, grad, 10);
**The result (random W) looks like**:
numerical: 2.822595 analytic: 2.825631, relative error: 5.375324e-04
numerical: -3.377733 analytic: -3.369572, relative error: 1.209526e-03
numerical: 0.228919 analytic: 0.217673, relative error: 2.518286e-02
numerical: 4.473230 analytic: 4.479649, relative error: 7.169656e-04
numerical: 0.776112 analytic: 0.773036, relative error: 1.985387e-03
numerical: 3.098687 analytic: 3.106507, relative error: 1.260129e-03
numerical: 0.963028 analytic: 0.966172, relative error: 1.629399e-03
numerical: -0.099664 analytic: -0.118163, relative error: 8.492500e-02
numerical: -1.977065 analytic: -1.987397, relative error: 2.606005e-03
numerical: 0.145323 analytic: 0.141235, relative error: 1.426644e-02

Now that we have a naive implementation of the softmax loss function and its gradient, implement a vectorized version in softmax_loss_vectorized. The two versions should compute the same results, but the vectorized version should be much faster.

    tic;
    [loss_naive, grad_naive] = softmax_loss_naive(W, imdb.X_train, imdb.y_train, 0.00001);
    time = toc;
    fprintf('Naive loss: %e computed in %fs\n', loss_naive, time);

    tic;
    [loss_vectorized, grad_vectorized] = softmax_loss_vectorized(W, imdb.X_train, imdb.y_train, 0.00001);
    time = toc;
    fprintf('Vectorized loss: %e computed in %fs\n', loss_vectorized, time);

As we did for the SVM, we use the Frobenius norm to compare the two versions of the gradient.

    grad_difference = norm(grad_naive - grad_vectorized, 'fro');
    fprintf('Loss difference: %f\n', abs(loss_naive - loss_vectorized));
    fprintf('Gradient difference: %f\n', grad_difference);
**The result (random W) looks like**:
Naive loss: 2.368984e+00 computed in 10.286108s
Vectorized loss: 2.368984e+00 computed in 0.372336s
Loss difference: 0.000000
Gradient difference: 0.000000
    
## Stochastic Gradient Descent
**Firstly**, implement the function **linear_softmax_train.m** and **linear_softmax_predict.m** as we did for SVM.

Use the validation set to tune hyperparameters (regularization strength and learning rate). You should experiment with different ranges for the learning rates and regularization strengths; if you are careful you should be able to get a classification accuracy of over 0.35 on the validation set.

    learning_rates = [1e-7, 5e-7];
    regularization_strengths = [5e4, 1e8];

    results = zeros(length(learning_rates), length(regularization_strengths), 2); %a matrix resotres results
    best_val = -1;   %The highest validation accuracy that we have seen so far.
    best_softmax = struct(); %The Linears model that achieved the highest validation rate.
    
**TODO:** Use the validation set to set the learning rate and regularization strength. This should be identical to the validation that you did for the SVM; save the best trained softmax classifer in best_softmax.

Print out results.

    for i =1:length(learning_rates)
        for j= 1:length(regularization_strengths)
             fprintf('lr %e reg %e train accuracy: %f val accuracy: %f\n', ...
                 learning_rates(i), regularization_strengths(j), results(i,j,1), results(i,j,2));
        end
    end
    fprintf('best validation accuracy achieved during cross-validation: %f\n', best_val);
**The result (random W) looks like**:
lr 1.000000e-07 reg 5.000000e+04 train accuracy: 0.330898 val accuracy: 0.348000
lr 1.000000e-07 reg 1.000000e+08 train accuracy: 0.330020 val accuracy: 0.339000
lr 5.000000e-07 reg 5.000000e+04 train accuracy: 0.100265 val accuracy: 0.087000
lr 5.000000e-07 reg 1.000000e+08 train accuracy: 0.100265 val accuracy: 0.087000
best validation accuracy achieved during cross-validation: 0.348000

Evaluate the best softmax on test set

    y_test_pred = linear_softmax_predict(best_softmax, imdb.X_test);
    test_accuracy = mean(imdb.y_test == y_test_pred');
    fprintf('linear Softmax on raw pixels final test set accuracy: %f\n', test_accuracy);
**The result (random W) looks like**:
linear Softmax on raw pixels final test set accuracy: 0.350000

Visualize the learned weights for each class.

    %Depending on your choice of learning rate and regularization strength, these may
    %or may not be nice to look at.
    w = best_softmax.W(:,1:end-1); % strip out the bias
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
**The expected results**:
![](https://raw.githubusercontent.com/MatthiasDING/BE_1/master/Document/softmax_weights.jpg)