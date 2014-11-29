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
%visualize_digits(base*8);

%% NOW CLASSIFY
m_list = [1 2 4 6 8 10 20 30 40 50 100 200];
acc_vs_m = zeros(1,length(m_list));
k_val = zeros(1,length(m_list));
for i = 1:length(m_list)
    m = m_list(i);
    fprintf('PCA dim m: %d\n', m);
    X = double(collapse_image_matrix(tr_images));
    [base,ed,mean,projX] = pcaimg(X,m);    
    
    visualize_digits(base*10);
    title_string = strcat('PCA dims: ',int2str(m));
    title(title_string)
    train_set = projX;
    nfold = 10;
    
    for K=[1:10 15 20 35 50]
      nfold = 10;
      acc(K) = cross_validate_for_PCA(K, train_set, tr_labels, nfold, tr_identity);
      fprintf('%d-fold cross-validation with K=%d resulted in %.4f accuracy\n', nfold, K, acc(K));
    end
    [maxacc bestK] = max(acc);
    fprintf('K is selected to be %d.\n', bestK);
    
    k_val(i) = bestK;
    acc_vs_m(i) = maxacc;
end

%%
figure
[ax,p1,p2] = plotyy(m_list, k_val,  m_list, acc_vs_m);
ylabel(ax(1),'Best K') % label left y-axis
ylabel(ax(2),'Accuracy') % label right y-axis
xlabel(ax(2),'m') % label x-axis
set(p1,'Marker', 'o')
set(p2,'Marker', 'o')


%% NOW CLASSIFY
m_list = 35:45;
acc_vs_m = zeros(1,length(m_list));
k_val = zeros(1,length(m_list));
for i = 1:length(m_list)
    m = m_list(i);
    fprintf('PCA dim m: %d\n', m);
    X = double(collapse_image_matrix(tr_images));
    [base,ed,mean,projX] = pcaimg(X,m);    
    
    visualize_digits(base*10);
    title_string = strcat('PCA dims: ',int2str(m));
    title(title_string)
    train_set = projX;
    nfold = 10;
    
    for K=[1:10 15 20 35 50]
      nfold = 10;
      acc(K) = cross_validate_for_PCA(K, train_set, tr_labels, nfold, tr_identity);
      fprintf('%d-fold cross-validation with K=%d resulted in %.4f accuracy\n', nfold, K, acc(K));
    end
    [maxacc bestK] = max(acc);
    fprintf('K is selected to be %d.\n', bestK);
    
    k_val(i) = bestK;
    acc_vs_m(i) = maxacc;
end

%%
figure
[ax,p1,p2] = plotyy(m_list, k_val,  m_list, acc_vs_m);
ylabel(ax(1),'Best K') % label left y-axis
ylabel(ax(2),'Accuracy') % label right y-axis
xlabel(ax(2),'m') % label x-axis
set(p1,'Marker', 'o')
set(p2,'Marker', 'o')

%% NOW CLASSIFY
m_list = [5 10 20 30 40 50 60];
acc_vs_m = zeros(1,length(m_list));
k_val = zeros(1,length(m_list));
for i = 1:length(m_list)
    m = m_list(i);
    fprintf('PCA dim m: %d\n', m);
    X = double(collapse_image_matrix(tr_images));
    [base,ed,mean,projX] = pcaimg(X,m);    
    %kick out the first couple of dimensions
    %base = base(:,3:end);
    base = base(:,5:end);
    
    visualize_digits(base*10);
    title_string = strcat('PCA dims: ',int2str(m));
    title(title_string)
    train_set = projX;
    nfold = 10;
    
    for K=[1:10 15 20 35 50]
      nfold = 10;
      acc(K) = cross_validate_for_PCA(K, train_set, tr_labels, nfold, tr_identity);
      fprintf('%d-fold cross-validation with K=%d resulted in %.4f accuracy\n', nfold, K, acc(K));
    end
    [maxacc bestK] = max(acc);
    fprintf('K is selected to be %d.\n', bestK);
    
    k_val(i) = bestK;
    acc_vs_m(i) = maxacc;
end

%%
figure
[ax,p1,p2] = plotyy(m_list, k_val,  m_list, acc_vs_m);
ylabel(ax(1),'Best K') % label left y-axis
ylabel(ax(2),'Accuracy') % label right y-axis
xlabel(ax(2),'m') % label x-axis
set(p1,'Marker', 'o')
set(p2,'Marker', 'o')
