% Landmark positions
L1 = [5;5];
L2 = [-5;5];
% State Model Parameters
g_t = [1,0;0,1]
%Measurement Model parameters
z_t = [];

%Covariance
Rt = [0.1,0;0,0.1];
%Measurement Covariance
Qt = [0.5,0;0,0.5];

%Belief mean and sigma

mu=[];
sig=[];

mu_0 = [0;0] + randn(2,1);
sig_0 = [1,0;0,1];

sig_init = [];
sig_next = [];

% Timestep
dt = 0.5;
%Blank Velocity Vector
V = []

%Jacobian for state model
G_t = [1,0;0,1]

%initialize
sig_init = sig_0;
mu = mu_0

mu_list =[]; % list of beliefss
sig_list = []; %list of standard deviations for every belief
true_pose = []; %list of true poses based on motion model and noise
% start loop here 
x_t = [0;0]
n = 1;
for t=0:dt:40
    if t <= 10
        V = [1; 0];
    elseif t <= 20
        V = [0; -1];
    elseif t <= 30
        V = [-1; 0];
    else
        V = [0; 1];
    end

    %Predict Step
    init_pose = mu;
    next_pose = (g_t * init_pose) + dt*V; % updating mu with velocity model
    sig_next = G_t*sig_init*G_t' + Rt; % updating sigma with jacobian approach

    x_t = x_t + dt*V; %+ ((Rt)*randn(2, 1)); % calculating true pose with measurement noise
    true_pose(:, n) = x_t; % adding true pose to list of historical poses indexed by n

    %Update Step 
    z1 = norm(x_t - L1) + (Qt(1,1))*randn();
    z2 = norm(x_t - L2) + (Qt(2,2))*randn();
    z_t = [z1; z2];
    
    % Measurement Jacobian
    diff1 = next_pose - L1;
    d1 = norm(diff1);
    diff2 = next_pose - L2;
    d2 = norm(diff2);
    H_t = [diff1'/d1; diff2'/d2]; % calculated jacobian of measurement model

    h_pred = [norm(diff1); norm(diff2)];
    
    % calculating Kalman gain
    Kt = sig_next * H_t' / (H_t * sig_next * H_t' + Qt);
    
    % Update Mu and Sigma
    mu = next_pose + Kt * (z_t - h_pred);
    sig_init = (eye(2) - Kt * H_t) * sig_next;
    
    mu_list(:,n) = mu;
    sig_list(:,:,n)=sig_init;
    n=n+1;
end

hold on
% Plot true trajectory
plot(true_pose(1, :), true_pose(2, :), 'b-');

% Plot estimated trajectory
plot(mu_list(1, :), mu_list(2, :), 'r--');

plot(L1(1), L1(2), 'go', 'MarkerSize', 15, 'MarkerFaceColor', 'g', 'DisplayName', 'Landmark 1');
plot(L2(1), L2(2), 'mo', 'MarkerSize', 15, 'MarkerFaceColor', 'm', 'DisplayName', 'Landmark 2');
legend('True','Belief')

% Plot 3-sigma confidence bounds (ellipses at each timestep)
for k = 1:size(mu_list, 2)
    % Extract mean and covariance at timestep k
    mu_k = mu_list(:, k);
    sig_k = sig_list(:, :, k);
    
    % Compute eigenvalues and eigenvectors of covariance
    [V_eig, D] = eig(sig_k);
    
    % Create points on a circle
    theta = linspace(0, 2*pi, 100);
    
    % Create ellipse
    ellipse = 3 * [sqrt(D(1,1)) * cos(theta); 
                   sqrt(D(2,2)) * sin(theta)];
    
    ellipse = V_eig * ellipse;
    
    % Translate ellipse to mean position
    ellipse(1, :) = ellipse(1, :) + mu_k(1);
    ellipse(2, :) = ellipse(2, :) + mu_k(2);
    
    % Plot ellipse (light red color)
    if k == 1
        plot(ellipse(1, :), ellipse(2, :), 'r-', 'LineWidth', 0.5, ...
             'Color', [1 0.7 0.7], 'DisplayName', '3\sigma Confidence');
    else
        
        plot(ellipse(1, :), ellipse(2, :), 'r-', 'LineWidth', 0.5, ...
             'Color', [1 0.7 0.7], 'HandleVisibility', 'off');
    end
end