function [swarm, swarm_history, swarm_pos_t] = pso_min(F, x_min, x_max, T_max, n_p)
% F is function being minimized
% T_max is maximum iterations
% n_p is number of particles

d = numel(x_min); % dimension
velocity = rand(d, n_p) * 0.1;
swarm =  x_min + (x_max - x_min) .* rand(d, n_p); % particle position


informants = 3; % number of local best to use
swarm_best = inf([d, n_p, informants]); % best 

swarm_fitness = F(swarm);

swarm_history = zeros([size(swarm), T_max]);

phi = 2.07;
c1 = 1 / (phi - 1 + sqrt(phi^2 - 2 * phi));
c_max = phi * c1;

for t = 1:T_max
    c2 = rand*c_max;
    c3 = rand([1, 3]) * c_max;

    swarm_history(:, :, t) = swarm;

    swarm_fitness_new = F(swarm);
    for j=1:n_p
        if swarm_fitness_new(j) <= swarm_fitness(j)
            swarm_fitness(j) = swarm_fitness_new(j);
            
            informant_idx = F(swarm(:, j)) <= F(squeeze(swarm_best(:, j, :)));
            [~, j_max] = max(informant_idx); % first index
            swarm_best(:, j, j_max) = swarm(:, j); % best position in a particles history (pbest). local best unused
        end
    end

    [~, i_min] = min(swarm_fitness); % global best
    swarm_pose = swarm(:, i_min);


    for j=1:n_p
        vel = c1 * velocity(:, j) + c2 * (swarm_pose - swarm(:, j));
        for k = 1:informants
            if swarm_best(:, j, k) < inf
                vel = vel + c3(k) * (squeeze(swarm_best(:, j, k)) - swarm(:, j));
            end
        end

        swarm(:, j) = swarm(:, j) + vel ; 
        velocity(:, j) = vel;
    end
end
end
