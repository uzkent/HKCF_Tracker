function [weights] = assign_weights(particles, target)
% This function assigns weights to the particles
% It uses euclidean distance between the KCF observation and predicted
% position

% Compute euclidean distance and assign weight
distance = (target.x - particles(1,:)).^2 + (target.y - particles(2,:)).^2;
weights = exp( - 0.05 * distance );

% Normalize the weights
weights  = weights / sum(weights);