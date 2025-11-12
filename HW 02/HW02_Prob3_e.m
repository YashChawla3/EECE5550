% % Particle Filter
N = 1000;

Xt1 = cell(1,N); % input N Particles
Xt2 = cell(1,N);
for i = 1:N
    Xt1{i} = [0;0;0]; % assigning initial pose
end 

t1=0; %first time step



  t2 =10;%second time step
    
    
    
  phi_l = 1.5; % left wheel commanded angular velocity
  phi_r = 2; % right wheel commanded angular velocity

  r = 0.25; %wheel radius
  w = 0.5 ;%wheel track width

  sig_l = 0.05; % uncertainty in left wheel speed
  sig_r = 0.05; % uncertainty in right wheel speed
  sig_p = 0.10; % uncertainty in measurement speed
  Xi=[];

  dt = t2-t1;
    for i=1:1:N
    
        xi = Xt1{i}; % extracting particle from array
    
        x = xi(1);
        y = xi(2); % extracting X, Y and theta from particle
        angle = xi(3);
    
        T_x1 = [cos(angle),-sin(angle),x  %creating homogenous T for particle xi
                sin(angle),cos(angle),y
                0, 0, 1];
    
       % calculating motion model on lie group 
        phi_r_noise = phi_r + (sig_r*randn());
        phi_l_noise = phi_l + (sig_l*randn());
    
% converting lie group to euclidean space and updating particle
    % position using exp map 

        omega_dot = [0,-(r/w)*(phi_r_noise-phi_l_noise),(r/2)*(phi_r_noise+phi_l_noise)
                 (r/w)*(phi_r_noise-phi_l_noise),0,0
                 0,0,0];
    
        T_x2 = T_x1 * expm(dt*omega_dot); %updating particle position in SE(3) using exponential map

    % extracting X Y and theta from new particle pose and reassining it to updated particle vector 
        Xt2{i} = [T_x2(1,3);T_x2(2,3);atan2(T_x2(2,1),T_x2(1,1))];
     
    end

%% 


%Extract positions from Xt1 and Xt2
pos_t1 = [];
pos_t2 = [];

for i = 1:N
    pos_t1(i, :) = Xt1{i}(1:2)';
    pos_t2(i, :) = Xt2{i}(1:2)';

end

% Calculate statistics for both
mean_t1 = mean(pos_t1, 1)
mean_t2 = mean(pos_t2, 1)
cov_t1 = cov(pos_t1)
cov_t2 = cov(pos_t2)

% Plotting means with particle distributions

hold on;
plot(pos_t1(:,1), pos_t1(:,2), 'b.', 'MarkerSize', 8);
plot(mean_t1(1), mean_t1(2), 'k*', 'MarkerSize', 15, 'LineWidth', 2);
title('Initial Particles (Xt1)');

plot(pos_t2(:,1), pos_t2(:,2), 'r.', 'MarkerSize', 8);
plot(mean_t2(1), mean_t2(2), 'k+', 'MarkerSize', 15, 'LineWidth', 2);
title('Propagated Particles (Xt2)');
legend('Particles', 'Mean for T=0','Particles at T=10','Mean for T=10');
grid on
axis equal

