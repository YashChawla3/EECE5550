N = 1000;

Xt1 = cell(1,N); % input N Particles
Xtn = cell(5,1); % blank array to store particle poses at t=0 to t=20
for i = 1:N
    Xt1{i} = [0;0;0];  % assigning initial pose at t=0
end 
Xtn{1} = Xt1;

t1=0; %first time step

  %t2 =10;%second time step 
    
  phi_l = 1.5; % left wheel commanded angular velocity
  phi_r = 2; % right wheel commanded angular velocity

  r = 0.25; %wheel radius
  w = 0.5 ;%wheel track width

  sig_l = 0.05; % left wheel speed uncertainty
  sig_r = 0.05; % right wheel speed uncertainty
  sig_p = 0.10; %measurement uncertainty

  z = cell(1,4); % aarray containing measured positions at 4 different time steps

  z{1} = [1.6561;1.2847];
  z{2} = [1.0505;3.1059];
  z{3} = [-0.9875;3.2118];
  z{4} = [-1.645;1.1978];

  count = 2;
  z_count = 1;

  for t=5:5:20

      dt = t-t1;
      Xt2 = cell(1,N);
 
    for i=1:1:N
    
        xi = Xt1{i};
    
        x = xi(1);
        y = xi(2);
        angle = xi(3);
    
        T_x1 = [cos(angle),-sin(angle),x
                sin(angle),cos(angle),y
                0, 0, 1];
    
    
        phi_r_noise = phi_r + (sig_r*randn());
        phi_l_noise = phi_l + (sig_l*randn());
    
        omega_dot = [0,-(r/w)*(phi_r_noise-phi_l_noise),(r/2)*(phi_r_noise+phi_l_noise)
                 (r/w)*(phi_r_noise-phi_l_noise),0,0
                 0,0,0];
    
        T_x2 = T_x1 * expm(dt*omega_dot); % confirm order of multiplication
    
        Xt2{i} = [T_x2(1,3);T_x2(2,3);atan2(T_x2(2,1),T_x2(1,1))];
    end

%% 

% starting partcile filter sampling / importance 
wi=[1:N]; % array of probabilities
den = 2*sig_p^2;
diff = [1:N]; % empty array to store difference values 
for i = 1:1:N
    current_particle = Xt2{i};
    lt = current_particle(1:2,1);
    diff(i) = (norm(z{z_count} - lt))^2;  % measurement - predicted position
    wi(i) = (1/sqrt(den*pi)) * exp(-diff(i)/ den); % Changed expm to exp

end

cumulative=0;
for i = 1:1:N
    cumulative = cumulative + wi(i);
end

wi_weighted = wi/cumulative;
cdf = cumsum(wi_weighted);
X_bar = cell(1,N);
% resampling step 

% Generate systematic samples
u0 = rand() / N;  % Random starting point
u = u0 + (0:N-1)' / N;  % Equally spaced samples

% Resample
j = 1;
for i = 1:N
    while u(i) > cdf(j)
        j = j + 1;
    end
    X_bar{i} = Xt2{j};  % Copy selected particle
end
     
    Xtn{count} = X_bar;
   Xt1 = X_bar;
   t1 = t;
   count = count+1;
   z_count = z_count + 1;
  end


 
    %% Plotting Code
% Calculating Mean for every position
num_iters = 5;

for t = 1:num_iters
    coords = [N,2];

    % Extract positions
    for i = 1:N
        positions(i, :) = Xtn{t}{i}(1:2)'; % extracting X,Y position from parent array containing all iteration information 
    end

    % Calculate mean and covariance
    t
    mean_pos = mean(positions, 1)
    cov_pos = cov(positions)

end

times = [0, 5, 10, 15, 20];

for t = 1:num_iters
    coords = [N,2];

    % Extract positions
    for i = 1:N
        positions(i, :) = Xtn{t}{i}(1:2)'; % extracting X,Y position from parent array containing all iteration information 
    end

    % Calculate mean and covariance
    t;
    mean_pos = mean(positions, 1);
    cov_pos = cov(positions);

end

% Plot all particle sets on one plot
figure;
hold on;

colors = {'b', 'r', 'g', 'k', 'c'};
markers = {'.', '.', '.', '.', '.'};

for t = 1:num_iters
    positions = [N,2];

    % Extract positions
    for i = 1:N
        positions(i, :) = Xtn{t}{i}(1:2)';
    end

    % Plot particles
    plot(positions(:,1), positions(:,2), [colors{t}, markers{t}],'MarkerSize', 5, 'DisplayName', sprintf('t = %d s', times(t)));
end

xlabel('x (m)');
ylabel('y (m)');
title('Particle Filter: Measured and Filtered Positions');
legend('Location', 'best');
grid on;
axis equal;
hold off;
