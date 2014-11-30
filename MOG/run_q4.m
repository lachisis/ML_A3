%MOG

load labeled_images.mat;
load public_test_images.mat;

tr_fixed = double(collapse_image_matrix(tr_images));

n = size(tr_fixed,2);

errorTrain = zeros(1, 4);
errorValidation = zeros(1, 4);
errorTest = zeros(1, 4);
numComponent = [2,5,10,15,30,50];
iters = 10;
minVary = 0.01;
plotFlag = 0;
randconst = 1;
use_kmeans = 1;
nfolds = 4;

train_errors = zeros(length(numComponent),1);
valid_errors = zeros(length(numComponent),1);
test_errors = zeros(length(numComponent),1);
test_errors_current = zeros(nfolds,10,1);
train_errors_current = zeros(nfolds,10,1);
valid_errors_current = zeros(nfolds,10,1);
%%
%Classification using MoG, using different numbers of components

[cross_val_train_ids, cross_val_valid_ids] = cross_validate_indeces(tr_images, nfolds);
%%
for i = 1:length(numComponent)
    %% repeat this 10 times so we can get an average performance
    K = numComponent(i);
    fprintf('N components: %d\n', K);
    for j = 1:nfolds
        %%
        x_train = tr_fixed(:,cross_val_train_ids{j});
        l_train = tr_labels(cross_val_train_ids{j});
        x_valid = tr_fixed(:,cross_val_valid_ids{j});
        l_valid = tr_labels(cross_val_valid_ids{j});
        
        x_1 = x_train(:,l_train==1);
        x_2 = x_train(:,l_train==2);
        x_3 = x_train(:,l_train==3);
        x_4 = x_train(:,l_train==4);
        x_5 = x_train(:,l_train==5);
        x_6 = x_train(:,l_train==6);
        x_7 = x_train(:,l_train==7);    
        
        z_1 = x_valid(:,l_valid==1);
        z_2 = x_valid(:,l_valid==2);
        z_3 = x_valid(:,l_valid==3);
        z_4 = x_valid(:,l_valid==4);
        z_5 = x_valid(:,l_valid==5);
        z_6 = x_valid(:,l_valid==6);
        z_7 = x_valid(:,l_valid==7);    
        %%
        for i_run = 1:5
            fprintf('run: %d\n', i_run);
            
            % Train a MoG model with K components for digit 2
            %-------------------- Add your code here --------------------------------
                [p1,mu1,vary1,logProbX1] = mogEM_q3(x_1,K,iters,minVary,plotFlag,randconst,use_kmeans);
                [p2,mu2,vary2,logProbX2] = mogEM_q3(x_2,K,iters,minVary,plotFlag,randconst,use_kmeans);
                [p3,mu3,vary3,logProbX3] = mogEM_q3(x_3,K,iters,minVary,plotFlag,randconst,use_kmeans);
                [p4,mu4,vary4,logProbX4] = mogEM_q3(x_4,K,iters,minVary,plotFlag,randconst,use_kmeans);
                [p5,mu5,vary5,logProbX5] = mogEM_q3(x_5,K,iters,minVary,plotFlag,randconst,use_kmeans);
                [p6,mu6,vary6,logProbX6] = mogEM_q3(x_6,K,iters,minVary,plotFlag,randconst,use_kmeans);
                [p7,mu7,vary7,logProbX7] = mogEM_q3(x_7,K,iters,minVary,plotFlag,randconst,use_kmeans);
            % Caculate the probability P(d=1|x) and P(d=2|x), 
            % classify examples, and compute the error rate
            % Hints: you may want to use mogLogProb function
            %-------------------- Add your code here --------------------------------
            %%
            %training 2s


            [~, train_errors_out] = calculate_mog_error(p1,mu1,vary1,x_1,...
                                            p2,mu2,vary2,x_2,...
                                            p3,mu3,vary3,x_3,...
                                            p4,mu4,vary4,x_4,...
                                            p5,mu5,vary5,x_5,...
                                            p6,mu6,vary6,x_6,...
                                            p7,mu7,vary7,x_7);
            %%
            train_errors_current(j,i_run) = sum(train_errors_out)/n;

            [~, valid_errors_out] = calculate_mog_error(p1,mu1,vary1,z_1,...
                                            p2,mu2,vary2,z_2,...
                                            p3,mu3,vary3,z_3,...
                                            p4,mu4,vary4,z_4,...
                                            p5,mu5,vary5,z_5,...
                                            p6,mu6,vary6,z_6,...
                                            p7,mu7,vary7,z_7);
            valid_errors_current(j,i_run) = sum(valid_errors_out)/n;
        %do the validation set
        
         end
    end
    %%
    train_errors(i) = mean(mean(train_errors_current,2));
    valid_errors(i) = mean(mean(valid_errors_current,2));
    
    %average results
    %test_errors(i) = mean(test_errors_current);
    %valid_errors(i) = mean(valid_errors_current);
    %train_errors(i) = mean(train_errors_current);
end
%%
[~,best_K_ind] = min(valid_errors(2:end));
best_K = numComponent(best_K_ind);
%%
x_1 = tr_fixed(:,tr_labels==1);
x_2 = tr_fixed(:,tr_labels==2);
x_3 = tr_fixed(:,tr_labels==3);
x_4 = tr_fixed(:,tr_labels==4);
x_5 = tr_fixed(:,tr_labels==5);
x_6 = tr_fixed(:,tr_labels==6);
x_7 = tr_fixed(:,tr_labels==7);  
[p1,mu1,vary1,logProbX1] = mogEM_q3(x_1,K,iters,minVary,plotFlag,randconst,use_kmeans);
                [p2,mu2,vary2,logProbX2] = mogEM_q3(x_2,K,iters,minVary,plotFlag,randconst,use_kmeans);
                [p3,mu3,vary3,logProbX3] = mogEM_q3(x_3,K,iters,minVary,plotFlag,randconst,use_kmeans);
                [p4,mu4,vary4,logProbX4] = mogEM_q3(x_4,K,iters,minVary,plotFlag,randconst,use_kmeans);
                [p5,mu5,vary5,logProbX5] = mogEM_q3(x_5,K,iters,minVary,plotFlag,randconst,use_kmeans);
                [p6,mu6,vary6,logProbX6] = mogEM_q3(x_6,K,iters,minVary,plotFlag,randconst,use_kmeans);
                [p7,mu7,vary7,logProbX7] = mogEM_q3(x_7,K,iters,minVary,plotFlag,randconst,use_kmeans);
                
 
%% TEST SET
test_inputs_fixed = double(collapse_image_matrix(public_test_images));
    test_pred = calculate_mog_error_single(test_inputs_fixed,p1,mu1,vary1,...
                                    p2,mu2,vary2,...
                                    p3,mu3,vary3,...
                                    p4,mu4,vary4,...
                                    p5,mu5,vary5,...
                                    p6,mu6,vary6,...
                                    p7,mu7,vary7);
%%
% Print the predictions to file
if (length(test_pred) < 1253)
  test_pred = [test_pred'; zeros(1253-length(test_pred),1)];
end

fprintf('writing the output to prediction.csv\n');
fid = fopen('prediction.csv', 'w');
fprintf(fid,'Id,Prediction\n');
for i=1:length(test_pred)
  fprintf(fid, '%d,%d\n', i, test_pred(i));
end
fclose(fid);
%% Plot the error rate
% %-------------------- Add your code here --------------------------------
% figure
% plot(numComponent,train_errors, '.-',...
%     numComponent,valid_errors,'.-',...
%     numComponent,test_errors,'.-')
% legend('Training error', 'Validation error', 'Test error')
% ylabel('Classification error')
% xlabel('Number of components for each class')
% title('Classification error vs. Number of gaussian components')