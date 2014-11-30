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

%%
%visualize_digits(base*8);

%% NOW CLASSIFY
%m_list = [1 2 4 6 8 10 20 30 40 50 100 200];
m_list = 35:45;
acc_vs_m = zeros(1,length(m_list));
k_val = zeros(1,length(m_list));

X = double(collapse_image_matrix(tr_images));
n = size(X,2);
d = size(X,1);
[base,ed,mean_of_data,projX] = pcaimg(X,m_list(end));  
%%
for i = 1:length(m_list)
    m = m_list(i);
    fprintf('PCA dim m: %d\n', m);  
    
    %visualize_digits(base*10);
    %title_string = strcat('PCA dims: ',int2str(m));
    %title(title_string)
    projX = base(:,1:m)'*X;
    nfold = 10;
    K_list = [1:10 15 20 35 50];
    acc = zeros(1,length(K_list));
    %%
    for i_K=1:length(K_list)
      K = K_list(i_K);
      nfold = 10;
      [train_indeces, valid_indeces] = cross_validate_indeces(projX,nfold,tr_identity);
      acc_vs_folds = zeros(1,nfold);
      %%
      for i_fold = 1:nfold
          cv_tr_inputs = projX(:,train_indeces{i_fold});
          cv_tr_labels = tr_labels(train_indeces{i_fold});
          cv_va_inputs = projX(:,valid_indeces{i_fold});
          cv_va_labels = tr_labels(valid_indeces{i_fold});
          
          preds = knn_classifier_for_PCA(K, cv_tr_inputs, cv_tr_labels,cv_va_inputs)';
          acc_vs_folds(i_fold) = sum(preds' == cv_va_labels)/length(cv_va_labels);
      end
      %%
      acc(i_K) = mean(acc_vs_folds);
      %%
      fprintf('%d-fold cross-validation with K=%d resulted in %.4f accuracy\n', nfold, K, acc(i_K));
    end
    [maxacc bestK] = max(acc);
    fprintf('K is selected to be %d.\n', K_list(bestK));
    
    k_val(i) = K_list(bestK);
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

%%
for i = 1:length(m_list)
    m = m_list(i);
    fprintf('PCA dim m: %d\n', m);  
    
    %visualize_digits(base*10);
    %title_string = strcat('PCA dims: ',int2str(m));
    %title(title_string)
    
    nfold = 10;
    K_list = [1:10 15 20 35 50];
    acc = zeros(1,length(K_list));
    %%

      nfold = 10;
      [train_indeces, valid_indeces] = cross_validate_indeces(X,nfold,tr_identity);
      acc_vs_folds = zeros(1,nfold);
      %%
      for i_fold = 1:nfold
         
          cv_tr_inputs = X(:,train_indeces{i_fold});
          cv_tr_labels = tr_labels(train_indeces{i_fold});
          cv_va_inputs = X(:,valid_indeces{i_fold});
          cv_va_labels = tr_labels(valid_indeces{i_fold});
          
          [base,ed,mean_of_data,projX] = pcaimg(cv_tr_inputs,m); 
          tr_proj = projX;
          va_proj = base'*cv_va_inputs;
           for i_K=1:length(K_list)
             K = K_list(i_K);
             preds = knn_classifier_for_PCA(K, tr_proj, cv_tr_labels,va_proj)';
             acc_vs_K(i_K) = sum(preds' == cv_va_labels)/length(cv_va_labels);
             fprintf('%d-fold cross-validation with K=%d resulted in %.4f accuracy\n', nfold, K, acc_vs_K(i_K));
           end
          
          
      end
      %%
      acc_vs_folds(i_fold) = mean(acc_vs_K);
      %%
      
    
    [maxacc bestK] = max(acc);
    fprintf('K is selected to be %d.\n', K_list(bestK));
    
    k_val(i) = K_list(bestK);
    acc_vs_m(i) = maxacc;
end

%% CHOOSE m = 37 w/ k = 6
load public_test_images.mat
m = 37;
k = 7;
nfolds = 10;
X = double(collapse_image_matrix(tr_images));
[base,ed,mean_of_data,projX] = pcaimg(X,m);

test_images = double(collapse_image_matrix(public_test_images));
proj_X = base'*proj_X;
pred = knn_classifier_for_PCA(k, X, tr_labels, proj_X);
%%

if (length(pred) < 1253)
  pred = [pred; zeros(1253-length(pred), 1)];
end

% Print the predictions to file
fprintf('writing the output to prediction.csv\n');
fid = fopen('prediction.csv', 'w');
fprintf(fid,'Id,Prediction\n');
for i=1:length(pred)
  fprintf(fid, '%d,%d\n', i, pred(i));
end
fclose(fid);
