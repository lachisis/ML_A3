% Performs nfold cross-validation using tr_images/tr_labels on a parameter K
% If tr_identity is provided, uses that to do the split of the training images
% If tr_identity is not provided uses random permutations (disregards similar faces, bias in the training data)

function [cross_val_train_ids, cross_val_valid_ids, test_ids] = cross_validate_indeces_with_test_set(tr_images, nfold, tr_identity)

ntr = size(tr_images, 2);

if (~exist('tr_identity', 'var'))
  % random permutation (disregards similar faces)
  perm = randperm(ntr); 

  testsize = floor(ntr/(nfold + 1));
  ntronly = ntr - testsize;
  foldsize = floor(ntronly/nfold);
  for i=1:nfold-1
    foldids{i} = (i-1)*foldsize+1:(i*foldsize);
  end
  foldids{nfold} = ((nfold-1)*foldsize+1):ntronly;
  test_ids{1} = ntronly + 1:ntr;
else
  % generally one uses random permutation to specify the splits, but because of the special structure of the dataset
  % we use the identity of poeple for this purpose.
  unknown = find(tr_identity == -1);
  tr_identity(unknown) = -(1:length(unknown));
  
  % finding people with the same identity
  [sid ind] = sort(tr_identity);
  [a b] = unique(sid);
  npeople = length(a);

  % separating out people with the same identity
  people = cell(npeople,1);
  people{1} = ind(1:b(1));
  for i=2:npeople
    people{i} = ind(b(i-1)+1:b(i))';
  end
  
  % shuffling people
  people = people(randperm(npeople));
  
  % dividing people into groups of roughly the same size but not necessarily
  testsize = floor(npeople/(nfold + 1));
  ntronly = npeople - testsize;
  foldsize = floor(ntronly/nfold);
  for i=1:nfold-1
    foldids{i} = [people{(i-1)*foldsize+1:(i*foldsize)}];
  end
  foldids{nfold} = [people{(nfold-1)*foldsize+1:ntronly}];
  test_ids{1} = [people{ntronly + 1:npeople}];
end

% perform nfold training and validation
cross_val_train_ids = cell(1,nfold);
cross_val_valid_ids = cell(1,nfold);
for i=1:nfold
  traini_ids = [foldids{[1:(i-1) (i+1):nfold]}];
  testi_ids = foldids{i};
  
  cross_val_train_ids{i} = traini_ids;
  cross_val_valid_ids{i} = testi_ids;
end

end