function [] = applyHomograpy(target,frameCount);
% This function applies homography to get the accumulated homograpy
target.Hcum = ones(3,3);
for i = 1:frameCount
    target.Hcum = target.Hcum .* target.H;
end

% Apply Accumulated Homography
target.x = target.x * target.Hcum(1,1) + target.y * target.Hcum(1,2) + target.Hcum(1,3);
target.y = target.x * target.Hcum(2,1) + target.y * target.Hcum(2,2) + target.Hcum(2,3);
target.z = target.x * target.Hcum(3,1) + target.y * target.Hcum(3,2) + target.Hcum(3,3);

target.x = target.x / target.z;
target.y = target.y / target.z;