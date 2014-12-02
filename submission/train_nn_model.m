function model = train_nn_model(labeled_data, labels, unlabeled_data)
labeled = double(collapse_image_matrix(labeled_data));
unlabeled = double(collapse_image_matrix(unlabeled_data));
pca_input = normalize_mean_var([labeled unlabeled]);
input = normalize_mean_var(labeled);

num_components = 40;
[base,ed,data_mean,projX] = pcaimg(pca_input,num_components);
input = base'*input;
num_hidden = 40;
num_epochs = 2000; 

% Automatic initialization
num_classes = max(labels);
clear weights;

results = train_nn_model_helper(input, labels,num_hidden,...
    num_classes, num_epochs);
model{1} = results{1}; %weights
model{2} = base; %PCA basis
end