function [c, pos, population, i_ci, gbest, gbest_fitness, gbest_fitness_history] = elephant_1d(x, I, plotdata, plotpos)
rng(0)
% x = normalize(x, 'range', [-1 1]);
if nargin < 2
    plotdata = false;
end
if nargin < 3
    plotpos = false;
end

[n, d] = size(x);
x_max = max(x)+0.1;
x_min = min(x)-0.1;
beta = 1.5;
alpha = 0.5;
max_iter = 30;
k = 3;
m = 5;
n_ci = 10;
n_e = m*n_ci;
 
population = repmat(x_min, [k, 1]) + rand(k, d, n_ci * m) .* (repmat(x_max, [k, 1]) - repmat(x_min, [k, 1]));
i_ci = ones(1, m*n_ci);
% herd index
for i = 2:m
    i_ci((i-1)*n_ci+1:i*n_ci) = i; 
end
x_min = repmat(x_min, [k, 1]);
x_max = repmat(x_max, [k, 1]);



colorv2 = rand([1, 3, k]);


% initial fitness calculation
[population, i_ci, population_fitness, i_c] = fitness(population, i_ci, x);
gbest = population(:, :, 1); 
gbest_fitness = population_fitness(1);
cbest = i_c(:, 1); % indices of best solution
gbest_fitness_history = zeros(1, max_iter);


for t=1:max_iter
    gbest_fitness_history(t) = gbest_fitness;
    
%     population_center = mean(population, 3);

    for ci = 1:m
            
        herd = population(:, :, i_ci==ci);
        population_center = mean(herd, 3);
        herd(:, :, 1) = herd(:, :, 1) + beta * (population_center - herd(:, :, 1));  % goes towards  center
        herd(:, :, end) = x_min + rand(size(x_min)) .* (x_max - x_min);
        for j = 2:n_ci
            herd(:, :, j) = herd(:, :, j) + alpha * rand * (herd(:, :, 1) - herd(:, :, j));
        end

        for j = 1:n_ci
            e = herd(:, :, j);
            e(e<x_min) = x_min(e < x_min);
            e(e>x_max) = x_max(e > x_max);
            herd(:, :, j) = e;
        end

        population(:, :, i_ci==ci) = herd;
    end
    
    [population, i_ci, population_fitness, i_c] = fitness(population, i_ci, x);

    if max(population_fitness) >= gbest_fitness
        gbest = population(:, :, 1);
        gbest_fitness = population_fitness(1);
        cbest = i_c(:, 1);
    end

end

c = cbest;
pos = gbest;
end

%% sorted fitness function and centroids
% i_c(j) is cluster indices using centroids in elephant(j)
function [population, i_ci, scores, i_c] = fitness(population, i_ci, x)
[n, d] = size(x);
[k, d, n_e] = size(population);
D = zeros(n, n_e); i_c = zeros(n, n_e); scores = zeros(1,n_e);
for j = 1:n_e
    [D(:, j), i_c(:, j)] = pdist2(population(:, :, j), x, 'squaredeuclidean', 'Smallest', 1);

    D2 = zeros([length(x), 1]);
    cl_scores = zeros(1, k);
    for ell = 1:k
        x_in_c = x(i_c(:, j)==ell, :);
        cl_center = mean(x_in_c);
        cl_scores(ell) = sum((x_in_c - cl_center).^2, 'all');
        
        D2(i_c(:, j)==ell) = sum((x_in_c-cl_center).^2, 2);
    end
    scores(j) = -sum(cl_scores);
end
scores(isnan(scores)) = -1;
[~, i_sort] = sort(scores, 'descend');
scores = scores(i_sort);
population = population(:, :, i_sort);
i_ci = i_ci(i_sort);
i_c = i_c(:, i_sort);
end