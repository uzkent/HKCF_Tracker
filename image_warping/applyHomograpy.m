function [] = applyHomograpy(target, frameCount)
% This function applies homography to get the accumulated homograpy
target.Hcum = ones(3,3);
for i = 1:frameCount
    target.Hcum = target.Hcum .* target.H;
end

% Apply Accumulated Homography to Previous Position
target.x = target.x * target.Hcum(1,1) + target.y * target.Hcum(1,2) + target.Hcum(1,3);
target.y = target.x * target.Hcum(2,1) + target.y * target.Hcum(2,2) + target.Hcum(2,3);
target.z = target.x * target.Hcum(3,1) + target.y * target.Hcum(3,2) + target.Hcum(3,3);
target.x = target.x / target.z;
target.y = target.y / target.z;

% % Apply Homography to Particle Filter
% particles(1,:) = particles(1,:) * target.Hcum(1,1) + particles(2,:) * ...
%      target.Hcum(1,2) + target.Hcum(1,3);
% particles(2,:) = particles(1,:) * target.Hcum(2,1) + particles(2,:) * ... 
%     target.Hcum(2,2) + target.Hcum(2,3);
% target.z = particles(1,:) * target.Hcum(3,1) + particles(2,:) * ...
%     target.Hcum(3,2) + target.Hcum(3,3);
% particles(1,:) = particles(1,:) / target.z;
% particles(2,:) = particles(2,:) / target.z;
