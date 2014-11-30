function separated_data = separate_data_into_classes(tr_images,tr_labels)

separated_data = cell(1,7);
for i = 1:7
    separated_data{i} = tr_images(:,tr_labels==i);
end