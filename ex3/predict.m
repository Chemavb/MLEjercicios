function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1); %numero de ejemplos
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
for i = 1:m
    a1 = X(i,:)'; %a1 = x
    a1 = [1; a1]; %Concatenamos el 1 al inicio
    
    z2 = Theta1*a1; %z2 es de 25x1
    a2 = sigmoid(z2); %a2 es el sigmoid de z2
    a2 = [1; a2]; %concatenamos 1 al inicio (a2 es de 26x1)
    
    z3 = Theta2*a2; %z3 es de 10x1
    a3 = sigmoid(z3); %a3 es el sigmoid de z3
    
    [A, idx] = max(a3);
    p(i) = idx;
end








% =========================================================================


end
