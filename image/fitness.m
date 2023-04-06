function [scores, C, D2, thetas] = fitness(centroids, x, f_name, options)
%FITNESS returns the f_name score for clustering x using centroids c
% Inputs:
%   centroids            : centroids
%   x                         : datapoints (rows)
%   f_name               : fitness function name (CH, Silhouette, SSE)
%
% Outputs:
%   scores  : fitness score
    arguments
        % name (size) type {rules} = default
        centroids;
        x;
        f_name string {mustBeMember(f_name, {'SSE', 'CH', 'img', 'Silhouette'})} = 'SSE';
        % options.option_name (size) type {rule} = default
        % call by function('option_name', option_value)
        options.sort = zeros(size(x));
    end
    
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
    thetas = 0;

    if strcmpi(f_name, 'img')
        lambda = 1e-5 * 1/max(x,[],'all'); % divergence to color matching ratio
        tau = .2; % weight average to update theta
        w1 = .01;
        w2 = .001;       
        S_omega = sum(D_full); 
        
        thetas = reshape(C-1, [sqrt(size(x, 1)),sqrt(size(x, 1)), size(C, ndims(C))]);        
        Sp_omega = w1 * S_omega(:, 1, :) .* (thetas) + w2 * S_omega(:, 2, :) .* (1 - thetas);
        Sm_omega = w1 * S_omega(:, 1, :) .* (thetas) - w2 * S_omega(:, 2, :) .* (1 - thetas);

        % probably an incorrectly implemented image segmentation
        S1 = psf2otf([1 -1], [256 256 n_e]);
        S2 = psf2otf([1; -1], [256 256 n_e]);
        x = reshape(x, [256 256]);

        thetas = reshape(thetas, [size(x), n_e]);
        p1 = real(ifftp(fftp(thetas) .* S2));
        p2 = real(ifftp(fftp(thetas) .* S1));
        div_p = real(ifftp(fftp(p2) .* S2)) + real(ifftp(fftp(p1) .* S1));
        div_p = div_p ./ max(abs(div_p), 1);
        
        theta_tilde = thetas + 1/2 * Sm_omega - lambda * div_p;

%         div_p = reshape(div_p, [numel(x), n_e]);
        p1 = reshape(p1, [numel(x), n_e]);
        p2 = reshape(p2, [numel(x), n_e]);
        Sp_omega = reshape(Sp_omega, [numel(x) n_e]);
        S3 = squeeze(1/2 * sum(Sp_omega, 1));

%         scores = -(lambda * sum(div_p) + S3);
        scores = -(lambda * sum(abs(p1) + abs(p2)) + S3);
        thetas = thetas + tau .* (theta_tilde - thetas);
        thetas(thetas > 1) = 1; thetas(thetas < 0) = 0;
        C = reshape(thetas, [numel(x), n_e]);
        return
    end

    for j = 1:n_e
     if strcmp(f_name, 'CH') % Calinski-Harabasz (not really)
            D2 = zeros([length(x), 1]);
            cl_scores = zeros(1, k);
            for ell = 1:k
                x_in_c = x(C(:, j)==ell, :);
                cl_center = mean(x_in_c);
                cl_scores(ell) = sum((x_in_c - cl_center).^2, 'all');
                
                D2(C(:, j)==ell) = sum((x_in_c-cl_center).^2, 2);
            end
%             scores(j) = -sum(D(:, j));
            scores(j) = -sum(cl_scores);


        elseif  strcmp(f_name, 'SSE')  % 
            m = zeros(d, k);
            sse = zeros(1, k);
            for q = 1:k
                m(:, q) = mean(x(C(:, j)==q), 1);
                x_q = x(C(:, j)==q, :)';
                sse(k) = norm(x_q - m(:, q))^2;
            end
            scores(j) = -sum(sse);
        elseif strcmp(f_name, 'Silhouette') % 
            s = silhouette(x, C(:, j), "sqEuclidean");
            s(isnan(s)) = -1;
            scores(j) = mean(s);
        end
    end
    D2 = 0;
    thetas = 0;
end

%% page-wise fft2
function F = fftp(x)
    F = pagetranspose(fft(pagetranspose(fft(x, [], 1)), [], 1));
end

%% page-wise ifft2
function F = ifftp(x)
    F = pagetranspose(ifft(pagetranspose(ifft(x, [], 1)), [], 1));
end


