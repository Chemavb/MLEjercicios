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

%We format y from 5000x1 to 5000x10
y1 = zeros(m, num_labels);

for i = 1:m
    label = y(i);
    y1(i, label) = 1;
end
%y1 cada fila es un ejemplo. (i fila k columna).

a1 = [ones(m,1) X];

z2 = Theta1*a1'; %25x401 x 5000x401'
a2 = sigmoid(z2); %25x5000
a2 = [ones(1,m); a2]; %26x5000

z3 = Theta2*a2; %10x26x26x5000
a3 = sigmoid(z3); %10x5000. Cada columna es la prediccion de un ejemplo i.
a3 = a3'; %5000x10. Cada fila es un ejemplo. (i fila k columna).

J = 0;

for i = 1:m
    for k = 1:num_labels
        J = J + (-y1(i,k)*log(a3(i,k)) - (1-y1(i,k))*log(1-a3(i,k)));
    end
end
J = J/m;


%% Regularized cost function %%

ParamTheta1 = 0;
ParamTheta2 = 0;

for j=1:size(Theta1,1) %niveles activacion capa 1
    for k=2:size(Theta1,2) %num_features
        ParamTheta1 = ParamTheta1 + Theta1(j,k)^2;
    end
end

for j=1:size(Theta2,1) %niveles activacion capa 1
    for k=2:size(Theta2,2) %num_features
        ParamTheta2 = ParamTheta2 + Theta2(j,k)^2;
    end
end

ParamRegularized = ParamTheta1 + ParamTheta2;
ParamRegularized = lambda*ParamRegularized/(2*m);


J = J + ParamRegularized;

%%% Obtaining Theta Gradient %%%
for t = 1:m
    a_1 = X(t, :)';
    a_1 = [1; a_1];
    %Activaciones
    z_2 = Theta1*a_1;
    z_2sigmoid = [1; z_2];
    a_2 = sigmoid(z_2);
    a_2 = [1;a_2];
    z_3 = Theta2*a_2;
    a_3 = sigmoid(z_3);
    
    delta_3 = zeros(num_labels,1);
    for k = 1:num_labels
        delta_3(k) = a_3(k) - y1(t,k);
    end
    
    delta_2 = (Theta2'*delta_3).*sigmoidGradient(z_2sigmoid); %z_2sigmoid es 26x1
    delta_2 = delta_2(2:end); %
    
    Theta1_grad = Theta1_grad + delta_2*a_1';
    Theta2_grad = Theta2_grad + delta_3*a_2';
    
end

    Theta1_grad = Theta1_grad/m; % 25x401
    Theta2_grad = Theta2_grad/m; % 10x26
    
    Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m)*Theta1(:, 2:end)
    Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m)*Theta2(:, 2:end)
    

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)]



end
