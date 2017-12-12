limit           = 200;    % maximum number of points to compute scores for
num_inactive    = 1e5;    % number of inactive compounds
num_queries     = 500;    % budget for queries
num_experiments = 20;     % number of repetitions of each experiment
verbose         = false;  % whether to print exhaustive information
                          % about each queery
max_lookahead   = 2;      % maximum number of steps to look ahead
k               = 100;    % number of nearest neighbors to use
prior_alpha     = 0.001;  % pseudocount of positive observations
prior_beta      = 1;      % pseudocount of negative observations

variables_to_save = {'num_inactive', 'num_queries', 'num_experiments', ...
                     'k', 'prior_alpha', 'prior_beta', 'fingerprint', ...
                     'results'};

data_directory    = 'precomputed/';
results_directory = 'results/';

num_proteins = 120;

train_inds = zeros(num_proteins, num_experiments);
results    = zeros(num_proteins, num_queries, num_experiments, max_lookahead);

% setup common problem struct fields
problem.num_classes = 2;
problem.num_queries = num_queries;
problem.verbose     = verbose;

for protein_ind = 1:num_proteins
  fprintf('processing protein #%i/%i ...\n', ...
          protein_ind, ...
          num_proteins);

  % load precomputed nearest neighbors
  filename = sprintf('%starget_%i_%s_nearest_neighbors_%i.mat', ...
                     data_directory, ...
                     protein_ind, ...
                     fingerprint, ...
                     num_inactive);
  load(filename);

  num_points = size(nearest_neighbors, 1);
  num_active = num_points - num_inactive;

  problem.points = (1:num_points)';

  % create label vector
  labels = ones(num_points, 1);
  labels(1:num_inactive) = 2;
  label_oracle = get_label_oracle(@lookup_oracle, labels);

  % limit to k-nearest neighbors
  nearest_neighbors = nearest_neighbors(:, 1:k)';
  similarities      =      similarities(:, 1:k)';

  % precompute sparse weight matrix
  row_index = kron((1:num_points)', ones(k, 1));
  weights = sparse(row_index, nearest_neighbors(:), similarities(:), ...
                   num_points, num_points);

  alpha = [prior_alpha, prior_beta];

  model = get_model(@knn_model, weights, alpha);
  model = get_model(@model_memory_wrapper, model);

  expected_utility = get_score_function(@search_expected_utility, model);

  probability_bound = get_probability_bound(@knn_probability_bound, ...
          weights, max(weights), alpha);

  selectors = cell(max_lookahead, 1);
  for i = 1:max_lookahead
    selectors{i} = get_selector(@limited_search_bound_selector, model, ...
                                probability_bound, i, num_exploit);
  end

  % generate positive training indices in a repeatable way
  rng(protein_ind);

  active_ind = find(labels == 1);
  train_inds(protein_ind, :) = active_ind(randperm(num_active, num_experiments));

  for experiment = 1:num_experiments
    fprintf(' ... running experiment %i of %i:\n', experiment, num_experiments);

    % create this training set
    train_ind       = train_inds(protein_ind, experiment);
    observed_labels = 1;

    for lookahead = 1:max_lookahead
      tic;
      fprintf(' ... ... trying lookahead %i ...\n', lookahead);

      score_function = get_score_function(@expected_utility_lookahead, ...
              model, expected_utility, selectors, lookahead);

      query_strategy = get_query_strategy(@argmax, score_function);

      [chosen_ind, chosen_labels] = active_learning(problem, train_ind, ...
              observed_labels, label_oracle, selectors{lookahead}, query_strategy);

      results(protein_ind, :, experiment, lookahead) = cumsum(chosen_labels == 1);

      elapsed = toc;
      fprintf(' ... ... done, found %i/%i positives, took %0.1fs.\n', ...
              results(protein_ind, end, experiment, lookahead), ...
              num_active - 1, ...
              elapsed);
    end
  end

end

filename = sprintf('%sresults_%s_%i.mat', ...
                   results_directory, ...
                   fingerprint, ...
                   num_inactive);

save(filename, variables_to_save{:});