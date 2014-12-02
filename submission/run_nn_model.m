function predictions = run_nn_model(model, input_data)
input = double(collapse_image_matrix(input_data));
input = normalize_mean_var(input);

weights = model{1};
base = model{2};
input = base'*input;

results = run_nn_model_helper(input,weights);
[~,max_inds] = max(results{1});
predictions = ind2vec(max_inds);
save('test.mat','predictions');
end