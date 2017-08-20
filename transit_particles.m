function [particles] = transit_particles(particles)
% This function transit the particles from the previous step to the current
% step
noise_velocity = 2;
noise_position = 2;

% Transit the particles
n_particles = size(particles,2);
particles(1,:) = particles(1,:) + particles(3,:) + ...
normrnd(0,noise_position,[n_particles 1])';
particles(2,:) = particles(2,:) + particles(4,:) + ...
normrnd(0,noise_position,[n_particles 1])';
particles(3,:) = particles(3,:) + ...
normrnd(0,noise_velocity,[n_particles 1])';
particles(4,:) = particles(4,:) + ...
normrnd(0,noise_velocity,[n_particles 1])';

% Handle Boundaries
temp = particles(1:2,:);
temp(temp>1500) = 1500;
temp(temp<1) = 1;
particles(1:2,:) = temp;
