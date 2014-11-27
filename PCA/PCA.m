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

m = 25;
size(tr_images)

%reshape this so that each of the 32 columns sits on top of each othr
X = double(collapse_image_matrix(tr_images));
[base,ed,mean,projX] = pcaimg(X,m);

%%
visualize_digits(base*8);

%% NOW CLASSIFY
for m = [10 20 30 40 50 100 200 300 400 500]
    fprintf('PCA dim m: %d\n', m);
    X = double(collapse_image_matrix(tr_images));
    [base,ed,mean,projX] = pcaimg(X,m);    
    
    train_set = projX;
    nfold = 10;
    
    for K=[1:10 15 20 35 50]
      nfold = 10;
      acc(K) = cross_validate_for_PCA(K, train_set, tr_labels, nfold, tr_identity);
      fprintf('%d-fold cross-validation with K=%d resulted in %.4f accuracy\n', nfold, K, acc(K));
    end
    [maxacc bestK] = max(acc);
    fprintf('K is selected to be %d.\n', bestK);
end


