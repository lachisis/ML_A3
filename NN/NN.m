%PCA

load labeled_images.mat;
%load public_test_images.mat;
%load hidden_test_images.mat;

h = size(tr_images,1);
w = size(tr_images,2);

% if ~exist('hidden_test_images', 'var')
%   test_images = public_test_images;
% else
%   test_images = cat(3, public_test_images, hidden_test_images);
% end
tr_images = collapse_image_matrix(tr_images);

%%
nfold = 10;
%% NOW CLASSIFY
for h = 50
    fprintf('Hidden units: %d\n', h);
    
    mean_acc = cross_validate_for_NN(h,tr_images,tr_labels,5);
    fprintf('Result: %f\n', mean_acc);
%     [maxacc bestK] = max(acc);
%     fprintf('K is selected to be %d.\n', bestK);
end


