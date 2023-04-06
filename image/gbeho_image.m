% test clustering for color values
rng(0)

%% initialization
clear
close all

colors = [[0 0.4470 0.7410]; [0.8500 0.3250 0.0980];...
    [0.4660 0.6740 0.1880]; [0.4940 0.1840 0.5560]; [0.9290 0.6940 0.1250]; ...
    [0.3010 0.7450 0.9330]; [0.6350 0.0780 0.1840];[0 0.4470 0.7410]; [0.8500 0.3250 0.0980];...
    [0.4660 0.6740 0.1880]; [0.4940 0.1840 0.5560]; [0.9290 0.6940 0.1250]; ...
    [0.3010 0.7450 0.9330]; [0.6350 0.0780 0.1840]; [0.8500 0.3250 0.0980];...
    [0.4660 0.6740 0.1880]; [0.4940 0.1840 0.5560]; [0.9290 0.6940 0.1250]; ...
    [0.3010 0.7450 0.9330]; [0.6350 0.0780 0.1840];[0 0.4470 0.7410]; [0.8500 0.3250 0.0980];...
    [0.4660 0.6740 0.1880]; [0.4940 0.1840 0.5560]; [0.9290 0.6940 0.1250]; ...
    [0.3010 0.7450 0.9330]; [0.6350 0.0780 0.1840]]; % from https://www.mathworks.com/help/matlab/creating_plots/specify-plot-colors.html

%% elephants
k = 2; % number of clusters
m = 2; % number of herds
n_ci = 15; % number of elephants per herd
n_e = m*n_ci; % population size
i_m = zeros(1, m); % best elephant index for each herd
i_ci = ones(n_e, 1); % index assigns elephants to herds
for j = 2:m
    i_ci(n_ci*(j-1)+1:n_ci*j) = j;
end

beta_min = 0.2; beta_max = 1.2;
T = 15; 
tol = 0.09; 
show = 9999; 
epsilon = rand*1e-7;


load T1Web.mat; % from math473 hw5 or maybe it was hw4
img = fm;
% img = imread('elephant.png');
% img = imread('deer.jpg');
% img = imread('iris.png');
y = double(im2gray(imresize(img, [256 256])));
x = y;

x = reshape(x, [256*256, 1]);
I = ones(size(x()));

[n, d] = size(x);

x_max = repmat(max(x), [k, 1])+1;
x_min = repmat(min(x), [k, 1])-1;

swarm_pos =  rand([k, d, n_e]);
fitness_history = zeros(n_e, 1); 

swarm_pos = swarm_pos .* repmat((max(x)-min(x)),k,1,n_e) + repmat(min(x),k,1,n_e);

psr = .5;
[swarm_fitness, c] = fitness(swarm_pos, x, 'img');
[swarm_pos, i_ci, swarm_fitness, c] = sort_swarm(swarm_pos, i_ci, swarm_fitness, c);
gworst = swarm_pos(:, :, end); gworst_fitness = swarm_fitness(end); % iteration worst
gbest = swarm_pos(:, :, 1); gbest_fitness = swarm_fitness(1); % global best
pbest = swarm_pos; pbest_fitness = swarm_fitness;
cbest = c(:, 1);


gbest_fitness_history = zeros([T, 1]); % plot the gbest best fitnesses
best_fitness_history = zeros([T, 1]); % plot the population's best fitness

for t = 1:T
    beta = beta_min + (beta_max - beta_min) * (1 - (t / T)^3)^2;
    alpha = abs(beta * sin(3*pi/2 + sin(beta * 3*pi/2)));

    for ci = 1:m
        i_m(ci) = find(i_ci==ci, 1);
        r = rand([1, 4]);
        n_ci = sum(i_ci==ci);
        herd_fitness = swarm_fitness(i_ci==ci);
        herd = swarm_pos(:, :, i_ci==ci);
        swarm_pos_old = swarm_pos;
        
        % update best
        if t / T < .5
            C_sigma = trnd(1, k, d);
            herd(:, :, 1) = herd(:, :, 1) + C_sigma;
%                 herd(:, :, 1) = herd(:, :, 1) + beta * (herd(:, :, 1) - mean(herd, 3));
        else
            herd(:, :, 1) = herd(:, :, randi([1, n_ci], 1)) + 2 * (rand(size(herd(:, :, 1))) - 0.5) .* (herd(:, :, randi([1, n_ci], 1)) - herd(:, :, randi([1, n_ci], 1)));
%                 herd(:, :, 1) = herd(:, :, 1) + beta * (herd(:, :, 1) - mean(herd, 3));
        end 
        
        % update elephants
        for j = 2:n_ci
            if rand < psr
                herd(:, :, j) = herd(:, :, j) + rand(size(herd(:, :, j))) .* ( (pbest(:, :, j) + gbest) / 2 - herd(:, :, j)) + rand(size(herd(:, :, j))) .* ((pbest(:, :, j) - gbest) / 2 - herd(:, :, j));
            else
                herd(:, :, j) = herd(:, :, j) + alpha * randn .* (herd(:, :, 1) - herd(:, :, j));
            end
        end

        for j = 1:n_ci
            elephant = herd(:, :, j);
            elephant(elephant<x_min) = x_min(elephant<x_min);
            elephant(elephant>x_max) = x_max(elephant>x_max);
            herd(:, :, j) = elephant;
        end
        swarm_pos(:, :, i_ci==ci) = herd;
    end

    % update worst
    for ci = 1:m
        K = (-1 + 2*rand([k, d])) .* exp(-2*t / T);
        herd = swarm_pos(:, :, i_ci==ci);
        herd_fitness = swarm_fitness(i_ci==ci);
        herd(:, :, end) = herd(:, :, end) + .1 * (x_max - x_min) .* rand([k, d]) + K;
        swarm_pos(:, :, i_ci==ci) = herd;
        if herd_fitness(end) < gworst_fitness
            gworst_fitness = herd_fitness(end);
            gworst = herd(:, :, end);
            swarm_fitness(i_ci==ci) = herd_fitness;
        end
    end

    [swarm_fitness, c] = fitness(swarm_pos, x, 'img');
    [swarm_pos, i_ci, swarm_fitness, ~, i_sort] = sort_swarm(swarm_pos, i_ci, swarm_fitness, c);
    pbest = pbest(:, :, i_sort); pbest_fitness = pbest_fitness(i_sort);

    for j = 1:n_e
        if rand < 1
            x_leo = LEO(swarm_pos_old, j, gbest, gworst, alpha, x_min, x_max);
            [x_leo_f, c2] = fitness(x_leo, x, 'img');
            if x_leo_f >= swarm_fitness(j)
                swarm_pos(:, :, j) = x_leo;
                swarm_fitness(j) = x_leo_f(1);
            end
            if swarm_fitness(j) >= gbest_fitness
                    gbest_fitness = swarm_fitness(j);
                    gbest = swarm_pos(:, :, j);
                    cbest = c2;
            end
        end
    end

    for ci = 1:m
        herd = swarm_pos(:, :, i_ci==ci);
        herd(:, :, 1) = gbest;
        swarm_pos(:, :, i_ci==ci) = herd;
    end

    [swarm_fitness, c] = fitness(swarm_pos, x, 'img');
    [swarm_pos, i_ci, swarm_fitness, c, i_sort] = sort_swarm(swarm_pos, i_ci, swarm_fitness, c);
    pbest = pbest(:, :, i_sort); pbest_fitness = pbest_fitness(i_sort);
    pbest(:, :, swarm_fitness>=pbest_fitness) = swarm_pos(:, :, swarm_fitness>=pbest_fitness);

    best_fitness_history(t) = swarm_fitness(1);
    gbest_fitness_history(t) = gbest_fitness;

    
    if mod(t, show) == 0
        [t -gbest_fitness]
    end
end
%% try to show image
close all
figure(1)

image(imresize(img, [256 256]));
hold on
bw = reshape((cbest>.5), [256 256]);
visboundaries(bw);
title('gbeho')
axis image

figure(2)
imagesc(reshape(cbest>.5, [256 256]), [0 1]);
axis image;
title('gbeho');
-gbest_fitness

figure(3)
[c, e, s, d] = kmeans(y(:), 2, 'display', 'iter', 'Distance', 'sqeuclidean', 'Start', 'uniform');
imagesc(reshape(c>1.5, [256 256]), [0 1]);
title('kmeans');
axis image;
return


function [swarm_pos, i_ci, swarm_fitness, c, i_sort] = sort_swarm(swarm_pos, i_ci, swarm_fitness, c)
    [swarm_fitness, i_sort] = sort(swarm_fitness, 'descend');
    swarm_pos = swarm_pos(:, :, i_sort);
    i_ci = i_ci(i_sort);
    c = c(:, i_sort);
end



