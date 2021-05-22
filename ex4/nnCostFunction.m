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

Y =  zeros(length(y),num_labels);
for i = 1:m
  Y(i,y(i)) = 1;
endfor

%Z_2 = sigmoid([ones(m,1),X]*Theta1');
%Z_3 = sigmoid([ones(size(Z_2,1),1),Z_2]*Theta2');
A_1 = [ones(m,1),X];
Z_2 = Theta1*A_1'; % 25 x 5000
A_2 = [ones(size(Z_2',1),1),sigmoid(Z_2)']; % 5000 x 26
Z_3 = Theta2*A_2'; % 10 x 5000
h = sigmoid(Z_3)'; %  5000 x 10

J = -sum(sum( Y.*log(h) + (1-Y).*log(1 - h)))./m;

theta_1 = Theta1(:,2:size(Theta1,2));
theta_2 = Theta2(:,2:size(Theta2,2));
R = lambda  * (sum( sum ( theta_1.^ 2 )) + sum( sum ( theta_2.^ 2 ))) / (2*m);

J += R;

% BackPropagation

for i = 1 : m
  A_1 = [1,X(i,:)];
  Z_2 = Theta1*A_1'; % 25 x 1
  A_2 = [1,sigmoid(Z_2)']; % 1 x 26
  Z_3 = Theta2*A_2'; % 10 x 1
  h_i = sigmoid(Z_3); % 10 x 1
  delta_3 = h_i - (Y(i,:))';% 10 x 1  % L = 3
  delta_2 = ((Theta2'*delta_3).*sigmoidGradient([1;Z_2]))(2:end);
  Theta2_grad += delta_3*A_2;
  Theta1_grad += delta_2*A_1;
    endfor
 Theta2_grad /= m;
 Theta1_grad /= m;
% -------------------------------------------------------------
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
