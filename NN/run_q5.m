% Choose the best mixture of Gaussian classifier you have, compare this
% mixture of Gaussian classifier with the neural network you implemented in
% the last assignment. 


% Train neural network classifier. The number of hidden units should be
% equal to the number of mixture components. 

% Show the error rate comparison.

%-------------------- Add your code here --------------------------------

%%
init_nn

%run it twice to get 1200 epochs
%Note that this code spits out graphs of the inputs to the hidden weights
%as training progresses for every 50th epoch
train_nn

%spew out stats for the test data classification
%test_nn

%% Comparison between MoG and NN
% 
% %Hardcoded output from question4
% test_error_mog = 0.0203;
% valid_error_mog = 0.0140;
% train_error_mog = 0.0012;
% 
% %output from the NN
% test_error_nn = test_CE;
% valid_error_nn = valid_CE;
% train_error_nn = train_CE;
% 
% fprintf('Classification error(percent)\tMoG model\tNeural Net\n');
% fprintf('Training error:\t\t\t\t\t%f\t%f\n',train_error_mog*100,train_error_nn*100);
% fprintf('Validation error:\t\t\t\t%f\t%f\n',valid_error_mog*100,valid_error_nn*100);
% fprintf('Training error:\t\t\t\t\t%f\t%f\n',test_error_mog*100,test_error_nn*100);

%% Visualize the hidden unit weights

%  visualize_digits_massive(W1, 'Weights into hidden units')
