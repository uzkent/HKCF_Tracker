function [particles] = initiate_particles(target)
% This function initiates the particle for the particle filter
% Determine the hyperparameters
 n_particles = 1000;
 velocity_std = 15;
 position_std = 15;
 n_dimension = 4;
 
 % Initiate the particles now
 particles = zeros(n_dimension,n_particles);
 particles(1,:) = target.x + randi([-position_std position_std], ... 
     [n_particles 1]);
 particles(2,:) = target.y + randi([-position_std position_std], ...
     [n_particles 1]);
 particles(3,:) = randi([-velocity_std velocity_std], [n_particles 1]);
 particles(4,:) = randi([-velocity_std velocity_std], [n_particles 1]);
 
 