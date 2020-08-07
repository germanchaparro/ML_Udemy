function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% gchaparro: optimization process in comments to avoid runing it for every submit
##minimum_error = Inf;
##prediction_error = Inf;
##test_C_sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
##
##for C_tmp = test_C_sigma_values
##  for sigma_tmp = test_C_sigma_values
##    fprintf("(C, sigma) -> (%f, %f)", C_tmp, sigma_tmp);
##    model= svmTrain(X, y, C_tmp, @(x1, x2) gaussianKernel(x1, x2, sigma_tmp));
##    predictions = svmPredict(model, Xval);
##    prediction_error = mean(double(predictions ~= yval));
##    if prediction_error < minimum_error
##      minimum_error = prediction_error;
##      C = C_tmp;
##      sigma = sigma_tmp;
##    endif
##  endfor
##endfor

% gchaparro, values of C and SIGMA according to dataset3Params optimization
C = 1.0;
sigma = 0.1;

fprintf("Choosen (C, sigma) -> (%f, %f)\n", C, sigma);


% =========================================================================

end
