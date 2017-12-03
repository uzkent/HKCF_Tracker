function [] = display_particles(particles,pr)
% This function overlays the particles onto the original frame
for i = 1:size(particles,2)
    plot(particles(2,i), particles(1,i), pr); 
end