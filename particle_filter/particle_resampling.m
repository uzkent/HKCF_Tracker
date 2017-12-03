function [particles, weights] = particle_resampling(weights, particles)
% This function replaces the weak particles with the dominant ones - so
% called importance resampling

% Find the Efficient Number of Samples
n_efficient = 1/sum(weights.^2);

% Number of Particles to be Resampled
N = 1000;
indexes = resample_particles(weights, N);

% Resample the particles
if (n_efficient) < 300
   particles = particles(:,indexes);
end
weights = 1 / size(particles,2);
%std(std(particles(3:4,:)))

% Redistribute velocities if the variance is too small