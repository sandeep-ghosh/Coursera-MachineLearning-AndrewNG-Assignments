function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%add the bias element (all 1) with X .. (becomes 5000*401)
A1 = [ones(size(X,1),1) X];


% calculate A2 (2nd layer) 5000*25 where theta1 -> 25*401
Z2 = (A1 * Theta1');
A2 = sigmoid(Z2);


% add bias element in layer 2 ... 5000*26
A2_1 = [ones(size(A2,1),1) A2];

% calculate final layer ... 5000*10 where theta2 -> 10*26
Z3 = (A2_1 * Theta2');
A3 = sigmoid(Z3);

% y is provided as 5000*1 matrix, need to transform it to 5000*10 matrix based 
% on output.. num_labels provided as number of labels, its a scalar value 
% that contains total number of labels if num_labels = 10 then
%, y == (num_labels)' would 
% provide 5000 * 10 matrix with appropriate matching values.

%first create a row vector with number labels
num_vals = 1:num_labels;

%reshape the new matrix

y_ex = (y == num_vals);

% y_ex is 5000*10 matrix, now we need to use A3 to calculate cost.

% ---------------------


J = (-1/m) * trace(y_ex' * log(A3) + (1 - y_ex)' * log(1 - A3));

% ----------------------

% For regularisation we need to square and add all theta values, without bias

J = J + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2,2)) + sum(sum(Theta2(:,2:end).^2,2)));


% -------------------------------------------------------------
%Backpropagation
% Calculate Theta without the bias part 
theta1 = Theta1(:, 2:size(Theta1,2));  %25*400
theta2 = Theta2(:, 2:size(Theta2,2));  %10*25

%Calculate delta3 5000*10
delta3 = A3 - y_ex;

%calculate delta2 5000*25
delta2 = delta3 * theta2 .* sigmoidGradient(Z2); % 5000 *25

%Calculate gradients
Theta1_grad = delta2' * A1;  % 25*401
Theta2_grad = delta3' * A2_1; % 10*26 

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

%regularisation
%need to add a column with zeros with theta1 and theta2 so that dimension mathces
%and it ignores bias part.
theta1 = [zeros(size(theta1,1),1),theta1]; % 25*401
theta2 = [zeros(size(theta2,1),1),theta2]; % 10*26
Theta1_grad = Theta1_grad + (lambda/m) * theta1;
Theta2_grad = Theta2_grad + (lambda/m) * theta2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
