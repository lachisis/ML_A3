% inputs must be d*n
% labels must be n*1, with values from 1:k
% weights is a 2-celled struct:
%       weights{1} is a h*(d+1) matrix
%       weights{2} is a k*(h+1) matrix
%           note: +1's represent biases
%
% d = dimensions
% n = number of training samples
% k = num_classes
% h = num_hidden

% results is a the error rate (percentage classified incorrectly)

function error_rate = evaluate_nn_classifier(inputs,labels, weights)

results = run_nn_classifier(inputs,weights);
output = results{1};

labels = ind2vec(labels');
n = length(inputs);
[~, prediction] = max(output);
error_rate = 1 - (sum(sum(labels & ind2vec(prediction))) / n);
end