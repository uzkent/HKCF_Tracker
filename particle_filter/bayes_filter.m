function [target, particles] = bayes_filter(particles, target)
% This function updates the results of the KCF with the PF result

% Transit the particles
[particles] = transit_particles(particles);

%Display the Particles
%display_particles(particles,'bx');
        
% Compute the Weights
[weights] = assign_weights(particles, target);

% Perform Importance Resampling
[particles, weights] = particle_resampling(weights, particles);

% Update the Position Estimation
target.x = round(sum(weights .* particles(1,:)));
target.y = round(sum(weights .* particles(2,:)));
