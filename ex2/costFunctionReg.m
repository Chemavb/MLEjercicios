function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sum = 0;
sum2 = 0;
z = 0;

%Parte 1 del coste J1
for i = 1:m
    z = X(i,:)*theta;
    sum = sum + (-y(i)*log(sigmoid(z)) - (1-y(i))*log(1-sigmoid(z)));
end
J1 = sum/m;

%Parte 2 del coste J2 (debido a lambda)
for j = 2:n
    sum2 = sum2 + theta(j)^2;
end
J2 = lambda*sum2/(2*m);

J = J1 + J2;

%Gradiente dJ(theta)/d(theta0)
sum1 = 0;
for i = 1:m
    z0 = X(i, :)*theta;
    sum1 = sum1 + (sigmoid(z0) - y(i))*X(i,1);
end
grad(1) = sum1/m;

%Gradiente generico DJ(theta)/d(thetaj)
for j = 2:n
    sum = 0; %Para cada j reiniciamos sumatorio
    for i = 1:m
        z = X(i, :)*theta;
        sum = sum + (sigmoid(z)-y(i))*X(i,j);
    end
    grad(j) = sum/m + lambda*theta(j)/m;
end



% =============================================================

end
