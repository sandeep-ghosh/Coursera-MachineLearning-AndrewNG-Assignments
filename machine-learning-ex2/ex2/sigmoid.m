function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% formula is 1/(1+e^(-z))
%Elementwise power with eulers number
g = exp(1).^(-z);

%Add one
g = g + 1;

%determine the fraction
g = 1 ./g;

% =============================================================

end
