function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    suma_error_por_xj = 0;
    suma_error_por_xj2 = 0;
    
    for i = 1:m
       error_por_xj = (theta'*X(i,:)' - y(i))*X(i, 1);
       suma_error_por_xj = suma_error_por_xj + error_por_xj;
    end
    
    for i = 1:m
       error_por_xj2 = (theta'*X(i,:)' - y(i))*X(i, 2);
       suma_error_por_xj2 = suma_error_por_xj2 + error_por_xj2;
    end
    
    theta(1) = theta(1) - (alpha/m)*suma_error_por_xj;
    theta(2) = theta(2) - (alpha/m)*suma_error_por_xj2;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
