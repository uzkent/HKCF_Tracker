function [coords] = boundary_handling(coords)
% This function handles the boundaries in HSI ROI sampling
% Handle Values Smaller Than Zero
if min(coords(1),1) < 0
   coords = coords + (- coords(1) + 1);
end
% Handle Values Larger Than Image Dimension
if max(coords(end),1500) > 1500
   coords = coords - (coords(end) - 1500);
end