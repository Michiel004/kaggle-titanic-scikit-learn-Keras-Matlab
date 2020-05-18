%% Machine Learning Online Class - Exercise 4 Neural Network Learning

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 32;  % 20x20 Input Images of Digits

hidden_layer_size1 = 64;   % 25 hidden units
hidden_layer_size2 = 33;   % 25 hidden units

num_labels = 2;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

                          
                          
global cvLos
cvLos = 1;
%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('normTitanicTestv2.mat');
load('normTitanicTrainv2.mat');
load('TitanicTestYv2.mat');
load('TitanicTrainYv2.mat');

normTitanicTestv2 = table2array(normTitanicTestv2);
normTitanicTrainv2 = table2array(normTitanicTrainv2);

TitanicTestYv2 = table2array(TitanicTestYv2);
TitanicTrainYv2 = table2array(TitanicTrainYv2);

%m = 1309;
m = size(normTitanicTrainv2, 1);

%% ================ Part 5: Sigmoid Gradient  ================
%  Before you start implementing the neural network, you will first
%  implement the gradient for the sigmoid function. You should complete the
%  code in the sigmoidGradient.m file.
%
%fprintf('\nEvaluating sigmoid gradient...\n')

%g = sigmoidGradient([-1 -0.5 0 0.5 1]);
%fprintf('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ');
%fprintf('%f ', g);
%fprintf('\n\n');

%fprintf('Program paused. Press enter to continue.\n');
%pause;


%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size1);
initial_Theta2 = randInitializeWeights(hidden_layer_size1, hidden_layer_size2);
initial_Theta3 = randInitializeWeights(hidden_layer_size2, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:); initial_Theta3(:)];


%% =============== Part 7: Implement Backpropagation ===============
%  Once your cost matches up with ours, you should proceed to implement the
%  backpropagation algorithm for the neural network. You should add to the
%  code you've written in nnCostFunction.m to return the partial
%  derivatives of the parameters.
%
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
%checkNNGradients;

%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 500);

%  You should also try different values of lambda
%lambda = 0.4;
lambda = 0;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction2Hiddenlayers(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size1, ...
                                   hidden_layer_size2, ...
                                   num_labels, normTitanicTrainv2, TitanicTrainYv2, lambda,normTitanicTrainv2,TitanicTrainYv2);
                               
%lambda
%,normTitanicTestv2,TitanicTestYv2

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[initial_nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(initial_nn_params(1:hidden_layer_size1 * (input_layer_size + 1)), ...
                 hidden_layer_size1, (input_layer_size + 1));

g1 =   (hidden_layer_size1 * (input_layer_size + 1) + 1);    
g2 = g1 +(hidden_layer_size2 * (hidden_layer_size1 + 1)-1);

Theta2 = reshape(initial_nn_params(g1:g2), ...
                 hidden_layer_size2, (hidden_layer_size1 +1 ));
            


Theta3 = reshape(initial_nn_params(g2 + 1:end), ...
                 num_labels, (hidden_layer_size2 + 1));
             


%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict2HiddenLayers(Theta1, Theta2, Theta3,normTitanicTestv2);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == TitanicTestYv2)));

cost = cost/2; 
%cvLos = cvLos;
plot(cost)
hold on 
plot(cvLos(1:length(cost)))
hold off
legend('training cost','CV cost')
title("Cost function")

Tp = 0;
Fn = 0;
Fp = 0;
Tn = 0;

%confusion(normTitanicTestv2,TitanicTestYv2) Deep learning Toolbox needed.
%roc(normTitanicTestv2,TitanicTestYv2)
length(pred)

for n = 1 : length(pred)
    
   if(pred(n) == 1) 
    if(TitanicTestYv2(n) == 1)
      Tp =  Tp +1;
    end
   
   if(TitanicTestYv2(n) == 2)
      Fn =  Fn +1;
   end
   end
   
   if(pred(n)== 2) 
   if(TitanicTestYv2(n)== 1)
      Fp =  Fp +1;
   end
   
   if(TitanicTestYv2(n) == 2)
      Tn =  Tn +1;
   end
   end
   
end
   
disp("Print Tp Fn Fp and Tn")
Tp
Fn
Fp
Tn

disp("Print Precision:")
Precision = Tp/(Tp+Fp)

disp("Print Precision")
Recal = Tp/(Tp+Fn)

disp("Print F1 score:")

f1Score = 2 * (Precision*Recal)/(Precision+Recal)


