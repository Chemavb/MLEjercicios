function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
q = size(theta,1); %number of theta parameters

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
sum = 0;
z = 0;

for i = 1:m
    z = theta'*X(i,:)';
    sum = sum + (-y(i)*log(sigmoid(z)) - (1-y(i))*log(1-sigmoid(z)));
end

J = sum/m; %Computado J.

%Computando gradiente generico:
for j = 1:q
    sum = 0; %Para cada j reiniciamos sumatorio
    for i = 1:m
        z = theta'*X(i, :)';
        sum = sum + (sigmoid(z) - y(i))*X(i, j);
    end
    grad(j) = sum/m;
end



% =============================================================

end
