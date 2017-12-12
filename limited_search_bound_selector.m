% a simple modification to active_search_bound_selector to return at
% most limit items (heuristically chosen to be those with the highest
% marginal probability)

function test_ind = limited_search_bound_selector(problem, train_ind, ...
          observed_labels, model, probability_bound, lookahead, limit)

  test_ind = unlabeled_selector(problem, train_ind, []);

  % find point with current maximum posterior probability
  probabilities = model(problem, train_ind, observed_labels, test_ind);
  [p_star, one_step_optimal_ind] = max(probabilities(:, 1));
  one_step_optimal_ind = test_ind(one_step_optimal_ind);

  % if we only look ahead one step, we only need to consider the
  % point with the maximum probability (it is optimal)
  if (lookahead == 1)
    test_ind = one_step_optimal_ind;
    return;
  end

  % We will need to calculate the expected l-step utility for two
  % points, and we create the required problem structure here.

  % For the selectors, we use this function recursively.
  selectors = cell(lookahead, 1);
  for i = 1:(lookahead - 1)
    selectors{i} = get_selector(@active_search_bound_selector, model, ...
                                probability_bound, i);
  end

  expected_utility = get_score_function(@search_expected_utility, model);

  % find the l-step expected utility of the point with current maximum
  % posterior probability
  p_star_expected_utility = expected_utility_lookahead(problem, ...
          train_ind, observed_labels, one_step_optimal_ind, model, ...
          expected_utility, selectors, lookahead) - ...
      search_utility([], [], observed_labels);

  % find the maximum (l-1)-step expected utility among the
  % currently unlabeled points
  one_fewer_step_optimal_utility = max(expected_utility_lookahead(problem, ...
          train_ind, observed_labels, test_ind, model, expected_utility, ...
          selectors, lookahead - 1)) - ...
      search_utility([], [], observed_labels);

  % find a bound on the maximum (l-1)-step expected utility after
  % one more positive observation
  one_fewer_step_utility_bound = expected_search_utility_bound(problem, ...
          train_ind, observed_labels, test_ind, probability_bound, ...
          lookahead - 1, 1);

  % Now a point with probability p can have l-step utility at most
  %
  %        p  * (1 + one_fewer_step_utility_bound  ) +
  %   (1 - p) *      one_fewer_step_optimal_utility
  %
  % and we use this to find a lower bound on p by asserting this
  % quantity must be greater than the l-step expected utility of
  % the point with current maximum probability.
  optimal_lower_bound = ...
      (p_star_expected_utility - one_fewer_step_optimal_utility) / ...
      (1 + one_fewer_step_utility_bound - one_fewer_step_optimal_utility);

  ind = (probabilities(:, 1) >= min(optimal_lower_bound, p_star));

  if (nnz(ind) > limit)
    [~, sorted_ind] = sort(probabilities(:, 1), 'descend');
    ind = sorted_ind(1:limit);
  end

  test_ind = test_ind(ind);

end
