% inputs must be d*n
% label_values must be n*1, each value an integer from 1:k
% training will continue for num_epoch steps
% weights is a 2-celled struct:
%       weights{1} is a h*(d+1) matrix
%       weights{2} is a k*(h+1) matrix
%           note: +1's represent biases
%
%   if weights is not provided, it will be given a random initialization
%
% d = dimensions
% n = number of training samples
% k = num_classes
% h = num_hidden
%
% results is a 3-celled vector
%       results{1} is the final weights
%       results{2} is the log_error_record, a 1*num_epochs matrix
%       results{3} is the rate_error_record, a 1*num_epochs matrix
function results = train_nn_classifier(inputs, label_values, num_hidden, num_classes, num_epochs, weights)
%% replace inputs (testing)
% inputs = train_images;
% label_values = train_labels;
% num_hidden = 1000;
% num_classes = 7;
% num_epochs = 20;
% epoch = 0;
% clear weights;
%% initialization

dimens = size(inputs,1);
n = size(inputs,2);
eps = 0.01;  %% the learning rate 
momentum = 0.8;   %% the momentum coefficient

labels = ind2vec(label_values'); % give labels the one-hot-encoding we want

% if weights not provided, initialize them randomly
if (~exist('weights', 'var'))
   %%% make random initial weights smaller, and include bias weights
    weights{1} = 0.1 * randn(num_hidden, dimens + 1);
    weights{2} = 0.1 * randn(num_classes, num_hidden + 1);
end
dW1 = zeros(size(weights{1}));
dW2 = zeros(size(weights{2}));

log_error_record = zeros(1, num_epochs);
rate_error_record = zeros(1, num_epochs);

%% training
for epoch = 1:num_epochs
   
    %fprintf('starting epoch %d\n',epoch);
    results = run_nn_classifier(inputs, weights);
    
    output = results{1};
    h_output = results{2}; 
   
    
    log_error_record(epoch) = -sum(sum(labels.*log(output)));
    
    [~, prediction] = max(output);
    one_hot_prediction = ind2vec(prediction);
    % ensure prediction is k*n
    one_hot_prediction = [one_hot_prediction ; zeros(num_classes-size(one_hot_prediction,1),n)];
    
    rate_error_record(epoch) = 1 - (sum(sum(labels & one_hot_prediction)) / n);
    
    % Compute deriv
    dE_by_dz = output - labels; % k*n

    % Backprop
    dE_by_dh_output = (weights{2}(:,1:num_hidden))' * dE_by_dz; % h*n
    dE_by_dh_input = dE_by_dh_output .* h_output .* (1 - h_output); % h*n

    % Gradients for weights
    dE_by_dW1 = dE_by_dh_input * double([inputs;ones(1,n)]') ; % h*(d+1)
    dE_by_dW2 = dE_by_dz * [h_output;ones(1,n)]' ; % k*(h+1)
    
    %%%%% Update the weights at the end of the epoch %%%%%%
    dW1 = momentum * dW1 - (eps / n) * dE_by_dW1; % h*(d+1)
    dW2 = momentum * dW2 - (eps / n) * dE_by_dW2; % k*(h+1)
    
    weights{1} = weights{1} + dW1;
    weights{2} = weights{2} + dW2;
end

results{1} = weights;
results{2} = log_error_record;
results{3} = rate_error_record;
end