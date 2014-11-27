function Y = collapse_image_matrix(X)
%takes a pxqxn matrix, where pxq is a single image and n is the number of
%images
%and turns it into a vxn matrix
    Y = reshape(X,size(X,1)*size(X,2),size(X,3));
end