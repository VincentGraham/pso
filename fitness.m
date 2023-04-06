function [scores, C, D2, thetas] = fitness(centroids, x, options)
%FITNESS returns the f_name score for clustering x using centroids c
% Inputs:
%   centroids            : centroids
%   x                         : datapoints (rows)
%   f_name               : fitness function name (CH, Silhouette, SSE)
%
% Outputs:
%   scores  : fitness score
    
    [k, d, n_e] = size(centroids, [1, 2, 3]); % n_e elephants, k clusters, d dimensions
    [n, ~] = size(x); % n data points

   %% assign to clusters
    D = zeros(n, n_e); C = zeros(n, n_e);
    D_full = zeros(n, k, n_e); C_full = zeros(n, k, n_e);
    counts = zeros(k, n_e);
    scores = zeros(n_e, 1);

    for j = 1:n_e
        % k smallest distances
        [D_j, c_j] = pdist2(centroids(:, :, j), x, 'squaredeuclidean', 'Smallest', k);
        D(:, j) = min(D_j, [], 1);
        C(:, j) = c_j(1, :);
        D_full(:, :, j) = D_j';
        C_full(:, :, j) = c_j';
        for dim = 1:size(c_j, 1)
            counts(dim, j) = sum(C(:, j)==dim);
        end
    end

    for j = 1:n_e
            D2 = zeros([length(x), 1]);
            cl_scores = zeros(1, k);
            for ell = 1:k
                x_in_c = x(C(:, j)==ell, :);
                cl_center = mean(x_in_c);
                cl_scores(ell) = sum((x_in_c - cl_center).^2, 'all');
                D2(C(:, j)==ell) = sum((x_in_c-cl_center).^2, 2);
            end
            scores(j) = -sum(cl_scores); %% !uses negative SSE!
    end
end