%PCA

load labeled_images.mat;
load public_test_images.mat;
%load hidden_test_images.mat;

h = size(tr_images,1);
w = size(tr_images,2);

if ~exist('hidden_test_images', 'var')
  test_images = public_test_images;
else
  test_images = cat(3, public_test_images, hidden_test_images);
end


[base,ed,mean,projX] = pcaimg(tr_images,m);