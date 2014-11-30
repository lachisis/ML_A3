% inputs must be d*n
% weights is a 2-celled struct:
%       weights{1} is a h*(d+1) matrix
%       weights{2} is a k*(h+1) matrix
%           note: +1's represent biases
%
% d = dimensions
% n = number of training samples
% k = num_classes
% h = num_hidden

% results is a 
%   results{1} is a k*n matrix, with raw output, NOT one-hot-encoded
%   results{2} is a h*n matrix, with the hidden unit outputs, which
%       is required for backpropogation

function results = run_nn_classifier(inputs, weights)

    n = size(inputs,2);
    h_input = weights{1} * double([inputs;ones(1,n)]);
    h_output = 1 ./ (1 + exp(-h_input));
    
    %input to output layer
    out_input = weights{2} * [h_output;ones(1,n)];
    %output of output layer
    output = exp(out_input)./ ...
        repmat( sum(exp(out_input),1), size(out_input,1),1);
    
    results{1} = output;
    results{2} = h_output;
end