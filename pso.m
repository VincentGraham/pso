%% see https://arxiv.org/pdf/1809.01942.pdf for parameter selection reasons
function [swarm_pos,gbest] = pso(x)
rng(0)

k = 3;
n_p = 50; %
iterations = 30;

[n, d] = size(x);

w  = 0.72; 
c1 = 1.49; 
c2 = 1.49; 

swarm_vel = rand(k, d, n_p) * 0.1; 
swarm_pos =  rand(k, d, n_p); 
swarm_best = zeros(k, d);

c = zeros(n, n_p); 

swarm_pos = swarm_pos .* repmat(max(x)-min(x),k,1,n_p) + repmat(max(x)-min(x),k,1,n_p);
swarm_fitness(1:n_p) = 10000;
gbest_fitness_history = zeros([iterations, 1]);


for i = 1:iterations
    Dd = zeros(n, k, n_p); % distance matrix 
    for p = 1:n_p % particle p
        for j = 1:k % centroid j
            D_j = zeros(n,  1);
            for ell = 1:n % datapoint ell
                D_j(ell) = norm(swarm_pos(j, :, p) - x(ell,:));
            end
            Dd(:, j, p) = D_j;
        end
    end

    for p = 1:n_p
        [~, idx] = min(Dd(:, :, p), [], 2);
        c(:, p) = idx;
    end
    
    delete(pc); delete(txt);
    pc = []; txt = [];
    
    %% fitness
    avg_fitness = zeros(n_p, 1);
    for p = 1:n_p
        [avg_fitness(p), ~] = fitness(swarm_pos(:, :, p), x);
        avg_fitness(p) = -avg_fitness(p);

        if avg_fitness(p) < swarm_fitness(p)
            swarm_fitness(p) = avg_fitness(p);
            swarm_best(:, :, p) = swarm_pos(:, :, p);
        end
    end
    
    [~, idx] = min(swarm_fitness);
    swarm_pose = swarm_pos(:, :, idx);

    gbest = swarm_pos(:, :, idx);
    
    %% update swarm
    r1 = rand; r2 = rand;
    for p = 1:n_p
        inertia = w * swarm_vel(:, :, p);
        cognitive = c1 * r1 * (swarm_best(:, :, p) - swarm_pos(:, :, p));
        social = c2 * r2 * (swarm_pose - swarm_pos(:, :, p));
        vel = inertia + cognitive + social;
        
        swarm_pos(:, :, p) = swarm_pos(:, :, p) + vel;
        swarm_vel(:, :, p) = vel;
    end
end

end