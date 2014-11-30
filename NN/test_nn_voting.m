% Test Parameters
input = collapse_image_matrix(tr_images);
labels = tr_labels;
identity = tr_identity;
n_hiddens = 600;
num_models = 5;
plotting_interval = 20;
max_epochs = 1000; %if not a multiple of plotting_interval, total epochs will be less

% Automatic initialization
nfold = num_models + 1;
n_classes = max(labels);
num_intervals = idivide(int32(max_epochs),plotting_interval);
weight_records = cell(num_models, num_intervals);
train_input = cell(num_models,1);
valid_input = cell(num_models,1);
train_labels = cell(num_models,1);
valid_labels = cell(num_models,1);
clear weights;

[cross_val_train_inds, cross_val_valid_inds] = cross_validate_indeces(input, nfold, identity);

% Training
for i = 1:num_models
   fprintf('starting training on model %d\n',i);
   
   train_input{i} = input(cross_val_train_inds{i});
   valid_input{i} = input(cross_val_valid_inds{i});
   train_labels{i} = labels(cross_val_train_inds{i});
   valid_labels{i} = labels(cross_val_train_inds{i});
   
   results = train_nn_classifier(train_input{i}, train_labels{i},n_hiddens,n_classes,plotting_interval);
   weight_records{i,1} = results{1};   
   for interval = 2:num_intervals
      results = train_nn_classifier(train_input{i}, train_labels{i},n_hiddens,n_classes,plotting_interval, weight_records{i,interval-1});
      weight_records{i,interval} = results{1};
   end
end

%%
% Validation
test_input = input(cross_val_train_inds{nfold});
test_labels = labels(cross_val_train_inds{nfold});
num_samples = length(test_labels);
result_votes = zeros(n_classes,num_samples,num_models);
accuracy = zeros(num_intervals,1);

for interval = 1:num_intervals
    for i=1:num_models
        results = run_nn_classifier(test_input,weight_records{i,interval});
        result_votes(:,:,i) = results{1};
    end
    average_output = mean(result_votes,3);
    [~,predictions] = max(average_output);
    accuracy(interval) = (sum(sum(ind2vec(test_labels') & ind2vec(predictions))) / num_samples);
end

figure(1)
title(sprintf('Results after averaging RAW OUTPUT from %d NNs',num_models));
ylabel('accuracy');
xlabel('epochs trained');
plot(plotting_interval:plotting_interval:max_epochs, accuracy)
