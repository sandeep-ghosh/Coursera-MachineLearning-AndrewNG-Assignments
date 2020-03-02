function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%add the bias element (all 1) with X

A1 = [ones(size(X,1),1) X];

% calculate A2 (2nd layer) 4000*25
A2 = sigmoid((A1 * Theta1'));

% add bias element in layer 2 ... 4000*26
A2_1 = [ones(size(A2,1),1) A2];

% calculate final layer ... 4000*10
A3 = sigmoid((A2_1 * Theta2'));


%%
%A3 is m * num_labels matrx, need to find highest of each row and map 
%appropriate values.


[M, I] = max(A3');
% I is 1*m row vetor with max indices for each exmple
p = I';

% =========================================================================


end
