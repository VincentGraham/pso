function x_leo = LEO(swarm_pos, n, gbest, gworst, alpha, x_min, x_max)
%LEO  returns an operator that moves a solution from local optima
%   beta_min = 0.2, beta_max = 1.2 as described in the paper
% Inputs:
%   swarm_pos : potential solutions
%   n             : solution index (swarm(:, :, n) = x_n)
%   gbest        : the (possibly global) best solution
%   gworst      : worst solution
%   [min, max]  : of the dataset X
%   t             : iteration
%   T             : max iterations
%
% Outputs:
%   x_leo        : local escaping operator applied to x
%
% 

    epsilon = .01*rand;
    rho1 = 2 * rand * alpha - alpha;
    rho2 = 2 * rand * alpha - alpha; 

    [k, d, N] = size(swarm_pos); dim = k*d; % solutions are k*d dimension
    x_n = reshape(swarm_pos(:, :, n), [dim, 1]); 
    x_best = reshape(gbest, [dim, 1]);
    x_worst = reshape(gworst, [dim, 1]);
    x_min = reshape(x_min, [dim, 1]);
    x_max = reshape(x_max, [dim, 1]);

    r = randperm(N, 4); % (17)
    x_r = reshape(swarm_pos(:, :, r), [dim, 4]); % pick 4 random solutions

    %% X1
    delta = 2 * rand * abs(mean(x_r, 2) - x_n); 
    step = ((x_best - x_r(:, 1)) + delta) / 2; 
    delta_x = rand([dim, 1]) .* abs(step); 

%     z_n = x_n - randn([dim, 1]) .* (2 * delta_x .* x_n) ./ (x_worst - x_best + epsilon); 
    z_n = x_n - (rho1 * randn([dim, 1]) .* (2 * delta_x .* x_n) ./ (x_best - x_worst + epsilon)) + rand * rho2 * (x_best - x_n); % (22)
    dm = rand * rho2 * (x_best - x_r(:, 1)); % (18, 21)
    yp = rand * (x_n + z_n) ./ 2 + rand * delta_x; 
    yq = rand * (x_n + z_n) ./ 2 - rand * delta_x;
    
    % the gsr uses randn in I. Ahmadianfar et al. (original GSR paper) but not in the elephant paper
    gsr = randn .* rho1 * (2 * delta_x .* x_n) ./ (yp - yq + epsilon); % (14)
    
    X1 = x_n - gsr + dm; 

    %% X2
    delta = 2 * rand * abs(mean(x_r, 2) - x_n); 
    step = ((x_best - x_r(:, 1)) + delta) / 2; 
    delta_x = rand([dim, 1]) .* abs(step); 

%     z_n = x_n - randn * (2 * delta_x .* x_n) ./ (x_worst - x_best + epsilon); % (22)
    z_n = x_best - (randn([dim, 1]) .* (2 * delta_x .* x_n) ./ (x_best - x_worst + epsilon)) + rand * rho2 * (x_r(:, 1) - x_r(:, 2)); % (22)
    yp = rand * (x_n + z_n) ./ 2 + rand * delta_x;
    yq = rand * (x_n + z_n) ./ 2 - rand * delta_x;
    dm = rand * rho2 * (x_r(:, 1) - x_r(:, 2)); 
    gsr = randn * rho1 * (2 * delta_x .* x_n) ./ (yp - yq + epsilon); 
    X2 = x_best - gsr + dm; 

    %% X3 and new solution x^{m+1}_n
    rho3 = 2 * rand([dim, 1]) .* alpha - alpha;
    X3 = x_n - rho3 .* (X2 - X1);
    ra = rand([dim, 1]); rb = rand([dim, 1]);
    x_leo = ra .* (rb .* X1 + (1 - rb) .* X2) + (1 - ra) .* X3; % (24)

    %% LEO
    pr = .5; % parameter
    if rand < pr
        f1 = -1 + 2*rand([dim, 1]); f2 = randn([dim, 1]);
        rho1 = 2 * rand([dim, 1]) * alpha - alpha;
        L1 = rand([dim, 1]) < .5; L2 = rand([dim, 1]) < .5;
        u1 = L1 * 2 .* rand([dim, 1]) + (1 - L1);
        u2 = L1 .* rand([dim, 1]) + (1 - L1); 
        u3 = L1 .* rand([dim, 1]) + (1 - L1); 

        kay = floor(unifrnd(1, length(swarm_pos))) + 1;

        x_k = reshape(swarm_pos(:, :, kay), [dim, 1]);
        x_rand = x_min + rand([dim, 1]) .* (x_max - x_min); 
        
        x_p = L2 .* x_k + (1 - L2) .* x_rand; 
        
        if u1 < .5
            x_leo = x_leo + f1 .* (u1 .* x_best - u2 .* x_p) + f2 .* rho1 .* (u3 .* (X2 -X1) + u2 .* (x_r(:, 1)- x_r(:, 2))) ./ 2;
        else
            x_leo = x_best + f1 .* (u1 .* x_best - u2 .* x_p) + f2 .* rho1 .* (u3 .* (X2 -X1) + u2 .* (x_r(:, 1)- x_r(:, 2))) ./ 2;
        end
    end

    x_leo = reshape(x_leo, [k, d]);
end



