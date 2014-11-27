%% To run this program:
%%   First run initBp
%%   Then repeatedly call runBp until convergence.

train_CE_list = zeros(1, num_epochs);
valid_CE_list = zeros(1, num_epochs);
train_CFE_list = zeros(1, num_epochs);
valid_CFE_list = zeros(1, num_epochs);

start_epoch = total_epochs + 1;


num_train_cases = size(inputs_train, 2);
num_valid_cases = size(inputs_valid, 2);

for epoch = 1:num_epochs
  % Fprop
  h_input = W1' * inputs_train + repmat(b1, 1, num_train_cases);  % Input to hidden layer.
  h_output = 1 ./ (1 + exp(-h_input));  % Output of hidden layer.
  if (mod(epoch-1,100) == 0)
      figure
     imagesc(h_input)
     colormap(gray)
  end
  logit = W2' * h_output + repmat(b2, 1, num_train_cases);  % Input to output layer.
  prediction = 1 ./ (1 + exp(-logit));  % Output prediction. (7xN matrix)

  [~, pred_max] = max(prediction);
  % Compute cross entropy
  train_CE = -mean(mean((target_train == 1)' .* log(prediction(1,:)) + ...
                        (target_train == 2)' .* log(prediction(2,:)) + ...
                        (target_train == 3)' .* log(prediction(3,:)) + ...
                        (target_train == 4)' .* log(prediction(4,:)) + ...
                        (target_train == 5)' .* log(prediction(5,:)) + ...
                        (target_train == 6)' .* log(prediction(6,:)) + ...
                        (target_train == 7)' .* log(prediction(7,:))));
  
  % Compute classification error
  target_ones = [(target_train == 1)';...
                (target_train == 2)'; ...
                (target_train == 3)'; ...
                (target_train == 4)'; ...
                (target_train == 5)'; ...
                (target_train == 6)'; ...
                (target_train == 7)'];
  train_CFE = 1 - sum(pred_max == target_train')/length(target_train);

  % Compute deriv
  dEbydlogit = prediction - target_ones;

  % Backprop
  dEbydh_output = W2 * dEbydlogit;
  dEbydh_input = dEbydh_output .* h_output .* (1 - h_output) ;

  % Gradients for weights and biases.
  dEbydW2 = h_output * dEbydlogit';
  dEbydb2 = sum(dEbydlogit, 2);
  dEbydW1 = inputs_train * dEbydh_input';
  dEbydb1 = sum(dEbydh_input, 2);

  %%%%% Update the weights at the end of the epoch %%%%%%
  dW1 = momentum * dW1 - (eps / num_train_cases) * dEbydW1;
  dW2 = momentum * dW2 - (eps / num_train_cases) * dEbydW2;
  db1 = momentum * db1 - (eps / num_train_cases) * dEbydb1;
  db2 = momentum * db2 - (eps / num_train_cases) * dEbydb2;

  W1 = W1 + dW1;
  W2 = W2 + dW2;
  b1 = b1 + db1;
  b2 = b2 + db2;

  %%%%% Test network's performance on the valid patterns %%%%%
  h_input = W1' * inputs_valid + repmat(b1, 1, num_valid_cases);  % Input to hidden layer.
  h_output = 1 ./ (1 + exp(-h_input));  % Output of hidden layer.
  logit = W2' * h_output + repmat(b2, 1, num_valid_cases);  % Input to output layer.
  prediction = 1 ./ (1 + exp(-logit));  % Output prediction.
  valid_CE = mean(mean((target_valid == 1)' .* log(prediction(1,:)) + ...
                        (target_valid == 2)' .* log(prediction(2,:)) + ...
                        (target_valid == 3)' .* log(prediction(3,:)) + ...
                        (target_valid == 4)' .* log(prediction(4,:)) + ...
                        (target_valid == 5)' .* log(prediction(5,:)) + ...
                        (target_valid == 6)' .* log(prediction(6,:)) + ...
                        (target_valid == 7)' .* log(prediction(7,:))));
  [~, pred_max] = max(prediction);
  valid_CFE = 1 - sum(pred_max == target_valid')/length(target_train);
  
  %%%%%% Print out summary statistics at the end of the epoch %%%%%
  total_epochs = total_epochs + 1;
  if total_epochs == 1
      start_error = train_CE;
  end
  train_CE_list(1, epoch) = train_CE;
  valid_CE_list(1, epoch) = valid_CE;
  train_CFE_list(1, epoch) = train_CFE;
  valid_CFE_list(1, epoch) = valid_CFE;  
  fprintf(1,'%d  Train CE=%f, Valid CE=%f, Train CFE=%f, Valid CFE=%f\n',...
            total_epochs, train_CE, valid_CE, train_CFE, valid_CFE);
end

clf; 
if total_epochs > min_epochs_per_plot
  epochs = [1 : total_epochs];
end

%%
%%%%%%%%% Plot the learning curve for the training set patterns %%%%%%%%%
train_errors(1, start_epoch : total_epochs) = train_CE_list;
valid_errors(1, start_epoch : total_epochs) = valid_CE_list;
train_errors_CFE(1, start_epoch : total_epochs) = train_CFE_list;
valid_errors_CFE(1, start_epoch : total_epochs) = valid_CFE_list;

%%
f = figure;
hold on, ...
  plot(epochs(1, 1 : total_epochs), train_errors(1, 1 : total_epochs), 'b'),...
  plot(epochs(1, 1 : total_epochs), valid_errors(1, 1 : total_epochs), 'g'),...
  legend('Train', 'Test'),...
  title('Cross Entropy vs Epoch'), ...
  xlabel('Epoch'), ...
  ylabel('Cross Entropy');
hold off

f2 = figure;
  hold on, ...
  plot(epochs(1, 1 : total_epochs), train_errors_CFE(1, 1 : total_epochs), 'b'),...
  plot(epochs(1, 1 : total_epochs), valid_errors_CFE(1, 1 : total_epochs), 'g'),...
  legend('Train', 'Test'),...
  title('Classification Error vs Epoch'), ...
  xlabel('Epoch'), ...
  ylabel('Percentage classification error (%/100)');
hold off