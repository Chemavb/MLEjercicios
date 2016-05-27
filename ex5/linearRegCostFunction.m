function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
n = size(theta);
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%No se hace esto puesto que X ya viene con la primera columna a 1. X = [ones(m,1) X] Añadimos columna con 1
htheta = X*theta; %matriz m x 1 (prediccion)
error = htheta - y; %matriz de m x 1 (error)
erroresCuadrado = error.^2;
sumaErroresCuadrado = sum(erroresCuadrado);
J1 = 1/(2*m) * sumaErroresCuadrado; %Primer termino de J
thetaRegularized = theta(2:end);
thetaRegularizedCuadrado = thetaRegularized.^2;
J2 = lambda/(2*m) * sum(thetaRegularizedCuadrado);

J = J1 + J2;

%%% Generacion de gradientes %%%
grad(1) = (1/m)*sum(error.*X(:,1));%Este es diferente a los demas.

for j=2:n
    grad(j) = ((1/m)*sum(error.*X(:,j))) + (lambda/m)*theta(j);
end






% =========================================================================

grad = grad(:);

end
