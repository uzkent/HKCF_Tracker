function [pr_curve] = precision_curve(target,results)
%This function computes the precision curve for the target of interest

%Read the Ground Truth File
voiGT = dlmread(['/Volumes/Burak_HardDrive/Moving_Platform_HSI/Ground_Truth/Ground_Truth_Files/' num2str(target.id) '_track.txt']);

%Initiate Precision Curve
pr_curve = zeros(1,50);

%Iterate GT
validFrame = 0;
for i = 1:size(results,1)

    index = find(voiGT(:,1)==results(i,3));
    
    if index
        %Compute Euclidean Distance
        dist(i) = sqrt(((results(i,2) - voiGT(index,2))^2+((results(i,1) - voiGT(index,3)+0)^2)));

        %Compute Precision
        for j = 1:50

           if (dist(i) <= j)
               pr_curve(j) = pr_curve(j) + 1;
           end

        end
        validFrame = validFrame + 1;
    end

end
%Normalize Precision Curve
pr_curve = pr_curve / validFrame;

%Print Run-Time Performance
pr_curve = [pr_curve mean(dist) mean(results(:,4))];

% figure(2);
% plot(1:50,pr_curve,'r','Linewidth',3);
% xlabel('Distance');
% ylabel('Precision');