function print_predictions(test_pred)
%test pred is a 1xn vector

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

end