facemap = [
    zeros(11,3) ones(11,10) zeros(11,6) ones(11,10) zeros(11,3) ;
    zeros(9,32);
    zeros(12,8) ones(12,16), zeros(12,8)];
facemap_flat = collapse_image_matrix(facemap);
