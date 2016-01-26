# k-Nearest Neighbor (kNN) exercise

Complete and run *Run_knn.m* in this exercise.You could run this script section by section. For more details see the assignments page on the course website.

The kNN classifier consists of two stages:

  * During training, the classifier takes the training data and simply remembers it
  * During testing, kNN classifies every test image by comparing to all training images and transfering the labels of the k most similar training examples
  * The value of k is cross-validated
  
In this exercise you will implement these steps and understand the basic Image Classification pipeline, cross-validation, and gain proficiency in writing efficient, vectorized code.

#### section 1: Load database and check its dimensions

	%% Load the raw CIFAR-10 data.
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
	
#### section 2: Visualize some examples from the dataset

	%We show a few examples of training images from each class.
	show_datasets(imdb);
	
#### section 3: Subsample the dataset and reshape the data

	%% Subsample the data for more efficient code execution in this exercise
	imdb = subsample_datasets(imdb);

	%% Reshape the image data into rows
	imdb.train_data = reshape(imdb.train_data, size(imdb.train_data,1), []);
	imdb.test_data = reshape(imdb.test_data, size(imdb.test_data,1), []);
    
    disp('Training data shape: ');
    disp(size(imdb.train_data));
    disp('Training labels shape: ');
    disp(size(imdb.train_labels));
    disp('Test data shape: ');
    disp(size(imdb.test_data));
    disp('Test labels shape: ');
    disp(size(imdb.test_labels));
    
#### section 4: Call *knn_train* to create a knn classifier model

	% Remember that training a kNN classifier is a noop: 
	% the Classifier simply remembers the data and does no further processing 
	model = knn_train(imdb.train_data, imdb.train_labels);
	
We would now like to classify the test data with the kNN classifier. Recall that we can break down this process into two steps:

1. First we must compute the distances between all test examples and all train examples.
2. Given these distances, for each test example we find the k nearest examples and have them vote for the label.

Lets begin with computing the distance matrix between all training and test examples. For example, if there are Ntr training examples and Nte test examples, this stage should result in a Nte x Ntr matrix where each element (i,j) is the distance between the i-th test and j-th train example.

First, open  *classifier/knn/knn_compute_distances_two_loops.m* that uses a (very inefficient) double loop over all pairs of (test, train) examples and computes the distance matrix one element at a time.


#### section 5: Todo: open *classifier/knn/knn_compute_distances_two_loops.m* and implement it

	%Test your implementation:
	dists_two = knn_compute_distances_two_loops(model, imdb.test_data);
	disp(size(dists_two))
**The expected results**: 500 5000
	
#### section 6: Todo: now implement the function *knn_predict_labels.m* and run the code below:

	% We use k = 1 (which is Nearest Neighbor).
	k = 1;
	test_labels_pred = knn_predict_labels(model, dists_two, k);
	num_correct = sum(sum(test_labels_pred == imdb.test_labels));
	num_test = length(imdb.test_labels);
	accuracy = double(num_correct)/num_test;
	fprintf('Got %d / %d correct => accuracy: %f\n',num_correct, num_test, accuracy);
	fprintf('You should expect to see approximately 27%% accuracy\n');
**The expected results**:
Got 137 / 500 correct => accuracy: 0.274000
You should expect to see approximately 27% accuracy.

####section 7: Now lets try out a larger k, say k = 5:

	k = 5;
	test_labels_pred = knn_predict_labels(model, dists_two, k);
	num_correct = sum(sum(test_labels_pred == imdb.test_labels));
	num_test = length(imdb.test_labels);
	accuracy = double(num_correct)/num_test;
	fprintf('Got %d / %d correct => accuracy: %f\n',num_correct, num_test, accuracy);
	fprintf('You should expect to see a slightly better performance than with k = 1.\n');
    
**The expected results**:
Got 139 / 500 correct => accuracy: 0.278000
You should expect to see a slightly better performance than with k = 1.
	
####section 8: Now lets speed up distance matrix computation by using partial vectorization with one loop. Implement the function *knn_compute_distances_one_loop.m* and run the code below:

	dists_one = knn_compute_distances_one_loops(model, imdb.test_data);

	% To ensure that our vectorized implementation is correct, we make sure that it
	% agrees with the naive implementation. There are many ways to decide whether
	% two matrices are similar; one of the simplest is the Frobenius norm. In case
	% you haven't seen it before, the Frobenius norm of two matrices is the square
	% root of the squared sum of differences of all elements; in other words, reshape
	% the matrices into vectors and compute the Euclidean distance between them.
    
	difference = norm(dists_two - dists_one, 'fro');
	fprintf('Difference was: %f\n', difference);
	if difference < 0.001
	fprintf( 'Good! The distance matrices are the same\n');
	else
	fprintf( 'Uh-oh! The distance matrices are different\n');
	end
**The expected results**:
Difference was: 0.000000
Good! The distance matrices are the same


####section 9: Now implement the fully vectorized version inside *knn_compute_distances_no_loops.m* and run the code

	dists_no = knn_compute_distances_no_loops(model, imdb.test_data);

	%check that the distance matrix agrees with the one we computed before:
	difference = norm(dists_two - dists_no, 'fro');
	fprintf('Difference was: %f\n', difference);
	if difference < 0.001
	fprintf( 'Good! The distance matrices are the same\n');
	else
	fprintf( 'Uh-oh! The distance matrices are different\n');
	end
**The expected results**:
Difference was: 0.000000
Good! The distance matrices are the same

####section 10: Let's compare how fast the implementations are
    tic;
    knn_compute_distances_two_loops(model, imdb.test_data);
    two_loop_time = toc;
    fprintf('Two loop version took %f seconds\n',two_loop_time);

    tic;
    knn_compute_distances_one_loops(model, imdb.test_data);
    two_loop_time = toc;
    fprintf('One loop version took %f seconds\n',two_loop_time);

    tic;
    knn_compute_distances_no_loops(model, imdb.test_data);
    two_loop_time = toc;
    fprintf('No loop version took %f seconds\n',two_loop_time);
    fprintf('you should see significantly faster performance with the fully vectorized implementation\n');

**The expected results**:
Two loop version took 65.902485 seconds
One loop version took 23.783070 seconds
No loop version took 0.353753 seconds
you should see significantly faster performance with the fully vectorized implementation
    
####section 11: Cross-validation
We have implemented the k-Nearest Neighbor classifier but we set the value k = 5 arbitrarily. We will now determine the best value of this hyperparameter with cross-validation.

	num_folds = 5;
	k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100];
	X_train_folds = {};
	y_train_folds = {};

	% ################################################################################
	% # TODO:                                                                        #
	% # Split up the training data into folds. After splitting, X_train_folds and    #
	% # y_train_folds should each be lists of length num_folds, where                #
	% # y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
	% # Hint: Look up the mat2cell function.                                #
	% ################################################################################
    
    **your code**
    
	% ################################################################################
	% #                                 END OF YOUR CODE                             #
	% ################################################################################
	% 
	% # A dictionary holding the accuracies for different values of k that we find
	% # when running cross-validation. After running cross-validation,
	% # k_to_accuracies[k] should be a list of length num_folds giving the different
	% # accuracy values that we found when using that value of k.

	k_to_accuracies = {};

	% ################################################################################
	% # TODO:                                                                        #
	% # Perform k-fold cross validation to find the best value of k. For each        #
	% # possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
	% # where in each case you use all but one of the folds as training data and the #
	% # last fold as a validation set. Store the accuracies for all fold and all     #
	% # values of k in the k_to_accuracies dictionary.                               #
	% ################################################################################
    
	**your code** 
    
	% ################################################################################
	% #                                 END OF YOUR CODE                             #
	% ################################################################################
    
    %  Print out the computed accuracies
    for k = 1:size(k_to_accuracies,1)
        for i = 1:size(k_to_accuracies, 2)
            fprintf('k = %d, accuracy = %f\n',k_choices(k), k_to_accuracies(k, i));
        end
    end
	
**The expected results**:
k = 1, accuracy = 0.263000
k = 1, accuracy = 0.257000
k = 1, accuracy = 0.264000
k = 1, accuracy = 0.278000
k = 1, accuracy = 0.266000
k = 3, accuracy = 0.239000
k = 3, accuracy = 0.249000
k = 3, accuracy = 0.240000
k = 3, accuracy = 0.266000
k = 3, accuracy = 0.254000
k = 5, accuracy = 0.248000
k = 5, accuracy = 0.266000
k = 5, accuracy = 0.280000
k = 5, accuracy = 0.292000
k = 5, accuracy = 0.280000
k = 8, accuracy = 0.262000
k = 8, accuracy = 0.282000
k = 8, accuracy = 0.273000
k = 8, accuracy = 0.290000
k = 8, accuracy = 0.273000
k = 10, accuracy = 0.265000
k = 10, accuracy = 0.296000
k = 10, accuracy = 0.276000
k = 10, accuracy = 0.284000
k = 10, accuracy = 0.280000
k = 12, accuracy = 0.260000
k = 12, accuracy = 0.295000
k = 12, accuracy = 0.279000
k = 12, accuracy = 0.283000
k = 12, accuracy = 0.280000
k = 15, accuracy = 0.252000
k = 15, accuracy = 0.289000
k = 15, accuracy = 0.278000
k = 15, accuracy = 0.282000
k = 15, accuracy = 0.274000
k = 20, accuracy = 0.270000
k = 20, accuracy = 0.279000
k = 20, accuracy = 0.279000
k = 20, accuracy = 0.282000
k = 20, accuracy = 0.285000
k = 50, accuracy = 0.271000
k = 50, accuracy = 0.288000
k = 50, accuracy = 0.278000
k = 50, accuracy = 0.269000
k = 50, accuracy = 0.266000
k = 100, accuracy = 0.256000
k = 100, accuracy = 0.270000
k = 100, accuracy = 0.263000
k = 100, accuracy = 0.256000
k = 100, accuracy = 0.263000

    % # plot the raw observations
    figure;
    hold on;
    for k = 1:length(k_choices)
        scatter(ones(1, size(k_to_accuracies,2))*k_choices(k), k_to_accuracies(k,:))
    end
    accuaracies_mean = mean(k_to_accuracies,2);
    accuaracies_std  = std(k_to_accuracies,0,2);
    errorbar(k_choices, accuaracies_mean, accuaracies_std);
    title('Cross-validation on k')
    xlabel('k')
    ylabel('Cross-validation accuracy')
    hold off;
**The expected results**:
![](https://raw.githubusercontent.com/MatthiasDING/BE_1/master/Document/knn_cross.jpg)

    % # Based on the cross-validation results above, choose the best value for k,   
    % # retrain the classifier using all the training data, and test it on the test
    % # data.
    best_k = 6;
    model = knn_train(imdb.train_data, imdb.train_labels);
    labels_pred = knn_predict(model, imdb.test_data, best_k);
    num_correct = sum(sum(labels_pred == imdb.test_labels));
    num_test = length(imdb.test_labels);
    accuracy = double(num_correct)/num_test;
    fprintf('Got %d / %d correct => accuracy: %f\n',num_correct, num_test, accuracy);
**The expected results**:
Got 141 / 500 correct => accuracy: 0.282000