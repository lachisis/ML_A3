% Test Parameters
% input = collapse_image_matrix(tr_images);
% labels = tr_labels;
% identity = tr_identity;
% n_hiddens = 600;
% num_models = 5;
% plotting_interval = 20;
% max_epochs = 1000; %if not a multiple of plotting_interval, total epochs will be less

% Automatic initialization
% nfold = num_models + 1;
% n_classes = max(labels);
previous_intervals = size(weight_records,2);
weight_records = [weight_records cell(num_models, num_intervals)];

train_accuracy = [train_accuracy ; zeros(num_intervals,1)];
% train_input = cell(num_models,1);
% valid_input = cell(num_models,1);
% train_labels = cell(num_models,1);
% valid_labels = cell(num_models,1);
% clear weights;

% [cross_val_train_inds, cross_val_valid_inds] = cross_validate_indeces(input, nfold, identity);

% Training
for i = 1:num_models
   fprintf('starting training on model %d\n',i);
   
%    train_input{i} = input(cross_val_train_inds{i});
%    valid_input{i} = input(cross_val_valid_inds{i});
%    train_labels{i} = labels(cross_val_train_inds{i});
%    valid_labels{i} = labels(cross_val_train_inds{i});
     
   for interval = 1:num_intervals
      results = train_nn_classifier(train_input{i}, train_labels{i}, ...
          n_hiddens,n_classes,plotting_interval, ...
          weight_records{i,previous_intervals + interval-1});
      weight_records{i,previous_intervals + interval} = results{1};
      train_accuracy(previous_intervals+interval) = ...
          1-results{3}(length(results{3}));
   end
end

%%
% Validation
result_votes = zeros(n_classes,num_samples,num_models);
accuracy = [accuracy ; zeros(num_intervals,1)];

for interval = 1:num_intervals
    for i=1:num_models
        results = run_nn_classifier(test_input,...
            weight_records{i,previous_intervals + interval});
        result_votes(:,:,i) = results{1};
    end
    average_output = mean(result_votes,3);
    [~,predictions] = max(average_output);
    accuracy(previous_intervals + interval) = ...
        (sum(sum(ind2vec(test_labels') & ind2vec(predictions))) / num_samples);
end
%%
figure(1)
x = plotting_interval:plotting_interval:((previous_intervals+num_intervals)*plotting_interval);
plot(x, accuracy, x, train_accuracy);
legend('validation set','training set');
title(sprintf('Results after averaging RAW OUTPUT from %d NNs',num_models));
ylabel('accuracy');
xlabel('epochs trained');