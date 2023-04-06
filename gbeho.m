% algorithm from the "elephant herding optimization" 
% and "Combined Elephant Herding Optimization Algorithm 
% with K-means for Data Clustering"
% cbest is clustering from gbest (best solution)
function [swarm_pos, gbest, cbest] = gbeho(x)
rng(0)

k = 3; % number of clusters
m = 3; % number of herds
n_ci = 10; % number of elephants per herd
n_e = m*n_ci; % population size
i_m = zeros(1, m); % best elephant index for each herd
i_ci = ones(n_e, 1); % index assigns elephants to herds
for j = 2:m
    i_ci(n_ci*(j-1)+1:n_ci*j) = j;
end

psr = .2; % parameter
beta_min = 0.2; beta_max = 1.2;
T = 30; % max iterations
tol = 0.09; % end if fitness is high enough (silhouette only?)


[n, d] = size(x);

% bound each centroid in a solution
x_min = repmat(min(x), [k, 1])-1;
x_max = repmat(max(x), [k, 1])+1;

swarm_pos =  rand([k, d, n_e]);

swarm_pos = swarm_pos .* repmat((max(x)-min(x)),k,1,n_e) + repmat(min(x),k,1,n_e);
% [~, swarm_pos(:, :, 1), ~] = kmeans(x, k); % isnt using k-means to initialize cheating?
% [i_ci, ~] = kmeans(reshape(swarm_pos, [n_e, k*d]), m); % i_ci is herd indices



[swarm_fitness, c] = fitness(swarm_pos, x);
[swarm_pos, i_ci, swarm_fitness, ~] = sort_swarm(swarm_pos, i_ci, swarm_fitness, c);
gworst = swarm_pos(:, :, end); gworst_fitness = swarm_fitness(end); % iteration worst
gbest = swarm_pos(:, :, 1); gbest_fitness = swarm_fitness(1); % global best
pbest = swarm_pos; pbest_fitness = swarm_fitness;
last_best = gbest;



gbest_fitness_history = zeros([T, 1]); % plot the gbest best fitnesses

% implementation of Y. Duan et al.
for t = 1:T
    beta = beta_min + (beta_max - beta_min) * (1 - (t / T)^3)^2;
    alpha = abs(beta * sin(3*pi/2 + sin(beta * 3*pi/2)));

    % update elephants
    for ci = 1:m
        i_m(ci) = find(i_ci==ci, 1);
        n_ci = sum(i_ci==ci);
        herd = swarm_pos(:, :, i_ci==ci);

        % update best
        if t / T < .1
            C_sigma = trnd(1, k, d);
            herd(:, :, 1) = herd(:, :, 1) + C_sigma; % cauchy random (34)
%                 herd(:, :, 1) = herd(:, :, 1) + beta * (herd(:, :, 1) - mean(herd, 3));
        else
%             herd(:, :, 1) = herd(:, :, randi([1, n_ci], 1)) + 2 * (rand(size(herd(:, :, 1))) - 0.5) .* (herd(:, :, randi([1, n_ci], 1)) - herd(:, :, randi([1, n_ci], 1)));
                herd(:, :, 1) = herd(:, :, 1) + beta * (mean(herd, 3) - herd(:, :, 1));

        end 

        for j = 2:n_ci
            if rand < psr
%                 herd(:, :, j) = herd(:, :, j) + rand(size(herd(:, :, j))) .* ( (pbest(:, :, j) + gbest) / 2 - herd(:, :, j)) + rand(size(herd(:, :, j))) .* ((pbest(:, :, j) - gbest) / 2 - herd(:, :, j));
                herd(:, :, j) = herd(:, :, j) + rand(size(herd(:, :, j))) .* ( (last_best + gbest) / 2 - herd(:, :, j)) + rand(size(herd(:, :, j))) .* ((last_best - gbest) / 2 - herd(:, :, j));

            else
                herd(:, :, j) = herd(:, :, j) + alpha * rand(size(herd(:, :, j))) .* (herd(:, :, 1) - herd(:, :, j));
            end
        end

        swarm_pos(:, :, i_ci==ci) = herd;
    end


    % update worst
    for ci = 1:m
        K = (-1 + 2*rand([k, d])) .* exp(-2*t / T);
        herd = swarm_pos(:, :, i_ci==ci);
        herd(:, :, end) = herd(:, :, end) + .1 * (x_max - x_min) .* rand([k, d]) + K;
        swarm_pos(:, :, i_ci==ci) = herd;
    end

    [swarm_fitness, c] = fitness(swarm_pos, x);
    [swarm_pos, i_ci, swarm_fitness, c, i_sort] = sort_swarm(swarm_pos, i_ci, swarm_fitness, c);
    pbest = pbest(:, :, i_sort); pbest_fitness = pbest_fitness(i_sort);

    for j = 1:n_e
        if rand < .5
            x_leo = LEO(swarm_pos, j, gbest, gworst, alpha, x_min, x_max);
            [x_leo_f, c2] = fitness(x_leo, x);
            if x_leo_f >= swarm_fitness(j)
                swarm_pos(:, :, j) = x_leo;
                swarm_fitness(j) = x_leo_f;
                c(:, j) = c2;
            end
        end
        if swarm_fitness(j) >= gbest_fitness
                gbest_fitness = swarm_fitness(j);
                gbest = swarm_pos(:, :, j);
                cbest = c(:, j);
        end
        if swarm_fitness(end) <= gworst_fitness
            gworst_fitness = swarm_fitness(end);
            gworst = herd(:, :, end);
        end
    end

    for j = 1:n_e
            elephant = swarm_pos(:, :, j);
            elephant(elephant<x_min) = x_min(elephant<x_min);
            elephant(elephant>x_max) = x_max(elephant>x_max);
            swarm_pos(:, :, j) = elephant;
    end


    [swarm_fitness, c] = fitness(swarm_pos, x);
    [swarm_pos, i_ci, swarm_fitness, c, i_sort] = sort_swarm(swarm_pos, i_ci, swarm_fitness, c);
    pbest = pbest(:, :, i_sort); pbest_fitness = pbest_fitness(i_sort);
    pbest(:, :, swarm_fitness>=pbest_fitness) = swarm_pos(:, :, swarm_fitness>=pbest_fitness);

    gbest_fitness_history(t) = gbest_fitness;

    % check if gbest is changing
%     if abs(last_fitness - swarm_fitness(1)) < tol
%         [t -gbest_fitness]
%         break
%     end

    last_best = swarm_pos(:, :, 1);

    
    if mod(t, 10) == 0
        [t -gbest_fitness] % print progress
    end
end

end

%% fitness/error functions
function [swarm_pos, i_ci, swarm_fitness, c, i_sort] = sort_swarm(swarm_pos, i_ci, swarm_fitness, c)
    [swarm_fitness, i_sort] = sort(swarm_fitness, 'descend');
    swarm_pos = swarm_pos(:, :, i_sort);
    i_ci = i_ci(i_sort);
    c = c(:, i_sort);
end


