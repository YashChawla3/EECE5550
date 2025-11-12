%% ICP Algorithm Implementation for 3D Point Cloud Registration

% Part (a): Estimate Correspondences
function C = EstimateCorrespondences(X, Y, t, R, dmax)
    X = readtable('pclX.txt')
    Y = readtable("pclY.txt")
    % Y: nY x d pointcloud matrix  
    % t: d x 1 translation vector
    % R: d x d rotation matrix
    % dmax: maximum admissible distance
    % Returns: C, a K x 2 matrix where each row is [i, j] correspondence
    
    nX = size(X, 1);
    C = [];
    
    for i = 1:nX
        xi = X(i, :)';  % Column vector
        
        % Transform xi: R*xi + t
        transformed_xi = R * xi + t;
        
        % Find closest point in Y
        distances = vecnorm(Y' - transformed_xi, 2, 1);
        [min_dist, j] = min(distances);
        
        % Check if distance is within threshold
        if min_dist < dmax
            C = [C; i, j];
        end
    end
end

% Part (b): Compute Optimal Rigid Registration (Horn's Method)
function [t_hat, R_hat] = ComputeOptimalRigidRegistration(X, Y, C)
    % X: nX x d pointcloud matrix
    % Y: nY x d pointcloud matrix
    % C: K x 2 correspondence matrix
    % Returns: t_hat (d x 1), R_hat (d x d)
    
    K = size(C, 1);
    
    if K == 0
        error('No correspondences found!');
    end
    
    % Extract corresponding points
    X_corr = X(C(:, 1), :);  % K x d
    Y_corr = Y(C(:, 2), :);  % K x d
    
    % Calculate centroids (Equation 4)
    x_bar = mean(X_corr, 1)';  % d x 1
    y_bar = mean(Y_corr, 1)';  % d x 1
    
    % Calculate deviations from centroids (Equation 5)
    X_prime = X_corr - x_bar';  % K x d
    Y_prime = Y_corr - y_bar';  % K x d
    
    % Compute cross-covariance matrix W (Equation 6)
    W = (Y_prime' * X_prime) / K;  % d x d
    
    % Singular Value Decomposition
    [U, ~, V] = svd(W);
    
    % Construct optimal rotation (Equation 7)
    d = size(X, 2);
    diag_matrix = eye(d);
    diag_matrix(d, d) = det(U * V');
    R_hat = U * diag_matrix * V';
    
    % Recover optimal translation (Equation 7)
    t_hat = y_bar - R_hat * x_bar;
end

% Part (c): ICP Algorithm
function [t_hat, R_hat, C, rmse_history] = ICP(X, Y, t0, R0, dmax, num_icp_iters)
    % X: nX x d pointcloud matrix
    % Y: nY x d pointcloud matrix
    % t0: d x 1 initial translation
    % R0: d x d initial rotation
    % dmax: maximum admissible distance
    % num_icp_iters: number of iterations
    % Returns: t_hat, R_hat, final correspondences C, and RMSE history
    
    t_hat = t0;
    R_hat = R0;
    rmse_history = zeros(num_icp_iters, 1);
    
    for iter = 1:num_icp_iters
        % Estimate correspondences
        C = EstimateCorrespondences(X, Y, t_hat, R_hat, dmax);
        
        fprintf('Iteration %d: Found %d correspondences\n', iter, size(C, 1));
        
        if isempty(C)
            warning('No correspondences found at iteration %d', iter);
            break;
        end
        
        % Compute optimal registration
        [t_hat, R_hat] = ComputeOptimalRigidRegistration(X, Y, C);
        
        % Calculate RMSE for monitoring
        X_corr = X(C(:, 1), :);
        Y_corr = Y(C(:, 2), :);
        K = size(C, 1);
        
        errors = zeros(K, 1);
        for k = 1:K
            transformed = R_hat * X_corr(k, :)' + t_hat;
            errors(k) = norm(Y_corr(k, :)' - transformed)^2;
        end
        rmse_history(iter) = sqrt(mean(errors));
        
        fprintf('  RMSE: %.6f\n', rmse_history(iter));
    end
    
    rmse_history = rmse_history(1:iter);
end

% Part (d): Main Script to Run ICP on Provided Data
function main_icp()
    % Load pointclouds
    X = load('pclX.txt');  % nX x 3
    Y = load('pclY.txt');  % nY x 3
    
    fprintf('Loaded pointcloud X: %d points\n', size(X, 1));
    fprintf('Loaded pointcloud Y: %d points\n', size(Y, 1));
    
    % Set parameters
    t0 = zeros(3, 1);      % Initial translation
    R0 = eye(3);           % Initial rotation (identity)
    dmax = 0.25;           % Maximum distance threshold
    num_icp_iters = 30;    % Number of iterations
    
    % Run ICP
    fprintf('\nRunning ICP algorithm...\n');
    [t_hat, R_hat, C, rmse_history] = ICP(X, Y, t0, R0, dmax, num_icp_iters);
    
    % Display results
    fprintf('\n=== RESULTS ===\n');
    fprintf('Estimated translation t_hat:\n');
    disp(t_hat);
    fprintf('Estimated rotation R_hat:\n');
    disp(R_hat);
    
    % Calculate final RMSE
    K = size(C, 1);
    X_corr = X(C(:, 1), :);
    Y_corr = Y(C(:, 2), :);
    
    final_rmse = 0;
    for k = 1:K
        transformed = R_hat * X_corr(k, :)' + t_hat;
        final_rmse = final_rmse + norm(Y_corr(k, :)' - transformed)^2;
    end
    final_rmse = sqrt(final_rmse / K);
    
    fprintf('Final RMSE: %.6f\n', final_rmse);
    fprintf('Number of correspondences: %d\n', K);
    
    % Transform entire pointcloud X
    X_transformed = (R_hat * X' + t_hat)';
    
    % Plot results
    figure('Position', [100, 100, 1200, 500]);
    
    % Plot 1: Co-registered pointclouds
    subplot(1, 2, 1);
    scatter3(Y(:, 1), Y(:, 2), Y(:, 3), 20, 'b', 'filled', 'DisplayName', 'Target (Y)');
    hold on;
    scatter3(X_transformed(:, 1), X_transformed(:, 2), X_transformed(:, 3), ...
             20, 'r', 'filled', 'DisplayName', 'Transformed Source (X)');
    xlabel('X'); ylabel('Y'); zlabel('Z');
    title('Co-registered Pointclouds');
    legend('Location', 'best');
    grid on;
    axis equal;
    view(3);
    
    % Plot 2: RMSE convergence
    subplot(1, 2, 2);
    plot(1:length(rmse_history), rmse_history, 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('RMSE');
    title('ICP Convergence');
    grid on;
end

% Run the main function
% Uncomment the line below to execute:
% main_icp();