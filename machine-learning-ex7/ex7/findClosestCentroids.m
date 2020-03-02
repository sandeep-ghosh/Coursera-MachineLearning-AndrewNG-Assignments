function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%loop over training set
for a = 1:size(X,1)
min_d = intmax;
c = 0;
%loop over centroids
	for b = 1:K
	%use matlab norm function to calculate Euclidean Distance between the vectors
	x_e = X(a,:);
	c_e = centroids(b,:);
	d = norm(x_e - c_e);
	
	if d < min_d 
		min_d = d;
		c = b;
	end
	
	end
	%At the end of iteration set minimum distance for every row
	idx(a,1) = c;
end





% =============================================================

end

