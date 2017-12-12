num_inactive = 100000;  % number of inactive proteins to randomly subselect
k            = 100;     % number of nearest neighbors to compute

rng('default');

data_directory        = 'processed/';
precomputed_directory = 'precomputed/';

fingerprints = {'ecfp4', 'gpidaph3', 'maccs'};

% choose subset of data to use
load([data_directory fingerprints{1} '/labels']);

inactive_class = max(labels);

inactive_ind = find(labels == inactive_class);
active_ind   = find(labels ~= inactive_class);

to_keep = [inactive_ind(randperm(numel(inactive_ind), num_inactive)); ...
           active_ind];

labels = labels(to_keep);

num_proteins = max(labels) - 1;

for fingerprint = fingerprints
  fprintf('processing fingerprint %s ...\n', fingerprint{:});

  load([data_directory fingerprint{:} '/features']);

  features = sparse(features(:, 2), features(:, 1), 1);

  features = features(:, to_keep);
  % remove features that are always zero
  features = features(any(features, 2), :);

  for protein_ind = 1:num_proteins
    tic;
    fprintf('  computing nearest neighbors for protein #%i/%i (%i actives) ... ', ...
            protein_ind, num_proteins, nnz(labels == protein_ind));

    filename = sprintf('%starget_%i_%s_nearest_neighbors_%i.mat', ...
                       precomputed_directory, ...
                       protein_ind, ...
                       fingerprint{:}, ...
                       num_inactive);

    if (exist(filename, 'file') > 0)
      fprintf('file already exists!\n');
      continue;
    end

    this_ind = (labels == inactive_class) | (labels == protein_ind);
    this_features = features(:, this_ind);
    this_features = this_features(any(this_features, 2), :);

    [nearest_neighbors, similarities] = jaccard_nn(this_features, k);

    save(filename, 'nearest_neighbors', 'similarities');

    elapsed = toc;
    if (elapsed < 60)
      fprintf('done, took %is.\n', ceil(elapsed));
    else
      fprintf('done, took %0.1fm.\n', ceil(elapsed / 6) / 10);
    end
  end

end
