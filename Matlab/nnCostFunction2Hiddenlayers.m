function [J grad] = nnCostFunctionv9(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size1, ...
                                   hidden_layer_size2, ...
                                   num_labels, ...
                                   X, y, lambda,normTitanicTestv2,TitanicTestYv2)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs cassification
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
Theta1 = reshape(nn_params(1:hidden_layer_size1 * (input_layer_size + 1)), ...
                 hidden_layer_size1, (input_layer_size + 1));

g1 =   (hidden_layer_size1 * (input_layer_size + 1) + 1);    
g2 = g1 +(hidden_layer_size2 * (hidden_layer_size1 + 1)-1);

Theta2 = reshape(nn_params(g1:g2), ...
                 hidden_layer_size2, (hidden_layer_size1 +1 ));
            

Theta3 = reshape(nn_params(g2 + 1:end), ...
                 num_labels, (hidden_layer_size2 + 1));

% Setup some useful variables
m = size(X, 1);

         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));
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

Matrix_X = [ones(m, 1) X];
L1 = sigmoid(Theta1 * Matrix_X');

Matrix_2 = [ones(m, 1) L1'];
L2 = sigmoid(Theta2 * Matrix_2'); 

Matrix_3 = [ones(m, 1) L2'];
hTheta = sigmoid(Theta3 * Matrix_3'); 

Vec = zeros(num_labels, m);
for i=1:m,
  Vec(y(i),i)=1;
end

J = (1/m) * sum (sum ((-Vec) .* log(hTheta) - (1-Vec) .* log(1-hTheta)));

Reg = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))+ sum(sum(Theta3(:,2:end).^2))) * (lambda/(2*m));

J = J + Reg;




for t = 1:m

    
  a1 = [1; X(t,:)'];
	z2 = Theta1 * a1; 
	a2 = [1; sigmoid(z2)];
    
	z3 = Theta2 * a2; 
	a3 = sigmoid(z3); 
    a3 = [1; sigmoid(z3)];
    
    z4 = Theta3 * a3; 
	a4 = sigmoid(z4);
    
	delta_4 = a4 - Vec(:,t); 
	
  z2=[1; z2]; 
  z3=[1; z3];
    
  delta_3 = (Theta3' * delta_4) .* sigmoidGradient(z3); 
  delta_3 = delta_3(2:end);
  delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2); 

	delta_2 = delta_2(2:end);
    

	Theta3_grad = Theta3_grad + delta_4 * a3';
    Theta2_grad = Theta2_grad + delta_3 * a2';
	Theta1_grad = Theta1_grad + delta_2 * a1';
    
end;


Theta3_grad = (1/m) * Theta3_grad + (lambda/m) * [zeros(size(Theta3, 1), 1) Theta3(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:)];
pred = predict2HiddenLayers(Theta1, Theta2, Theta3,normTitanicTestv2);

global cvLos
%test = 1 - mean(double(pred == TitanicTestYv2));
%cvLos = [cvLos,(sum(abs(pred - TitanicTestYv2)))/length(TitanicTestYv2)];

global Hulp
if Hulp == 0 
    %length(TitanicTestYv2)
    %sum(abs(pred - TitanicTestYv2))
    %mean(double(pred == TitanicTestYv2))
    los = (sum(abs(pred - TitanicTestYv2)))/length(TitanicTestYv2);
    
    %{
    if (cvLos(length(cvLos)) + 1  <= los  )
        los = cvLos(length(cvLos));
    end
    %}
    
    cvLos = [cvLos ;los];
    Hulp = 0;
else
    Hulp = 0;
end


end