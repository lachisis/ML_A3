function visualize_digits(data_matrix)
% Visualize digit images for examples in the data matrix.
%
% data_matrix should be a n_dimensions x n_examples matrix, each column is one
% example.
% 
% This is intended only to visualize a small number (say < 10) of digits.
%
figure;
if ((ndims(data_matrix) == 2) & (size(data_matrix,1) ~= size(data_matrix,2))) 
    data_matrix = unpack_image_matrix(data_matrix);
    size(data_matrix)
end

if(ndims(data_matrix) ~= 3)
    imshow(data_matrix);
else
    n_examples = size(data_matrix, 3);
    if(n_examples > 10)
        n_rows = ceil(double(n_examples)/10);
        for j = 1 : n_rows - 1
            for i = 1:10
            subplot(n_rows, 10, (j-1)*10 + i);
            d = data_matrix(:,:,(j-1)*10 + i);
            imshow(d);
%             if(j==1 && i==5)
%                 title(fig_title)
%             end
            end
        end
        for j = n_rows
            n_cols_in_last_row = n_examples - (n_rows-1)*10;
            for i = 1:n_cols_in_last_row
            subplot(n_rows, 10, (j-1)*10 + i);
            if i <= 10
                d = data_matrix(:,:,(j-1)*10 + i);
                imshow(d);
            end
            if n_cols_in_last_row < 5
                if(j == 1 && i == n_cols_in_last_row)
                    title(fig_title)
                end
            else
                if(j==1 && i==5)
                    title(fig_title)
                end    
            end
            end
        end
    else
        for i = 1 : n_examples
            subplot(1, n_examples, i);
            d = data_matrix(:,:,i);
            imshow(d);
        end
    end
end
