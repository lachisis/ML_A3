function x = normalize_mean_var(inputs)
%inputs = dxn
    % Subtract mean for each image
    tr_mu = mean(inputs);
    x = bsxfun(@minus, inputs, tr_mu);

    % Normalize variance for each image
    tr_sd = var(x);
    tr_sd = tr_sd + 0.01; % for extreme cases
    tr_sd = sqrt(tr_sd);
    x = bsxfun(@rdivide, x, tr_sd);  
end