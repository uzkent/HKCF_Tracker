function [psr] = psr_est(response)
%% This function estimates the peak to sidelobe ratio of the response map
% \param[in] response : response map
% \param[out] psr : peak-to-sidelobe-ratio value
max_peak = max(max(response)); % peak value
[rows, cols] = size(response);
sd_size = 8;

% Index of the response
[ix,iy] = find(response==max(max(response)));

% Make the pixels in the vicinity of the peak NaN
response(max(1,ix-sd_size):ix+(sd_size-1),...
    max(1,iy-sd_size):iy+(sd_size-1)) = NaN;
response = response(1:rows,1:cols);

mu = nanmean(nanmean(response)); % Mean of sidelobe
dev = nanstd(nanstd(response));    % Standard deviation of sidelobe

psr = (max_peak - mu) / dev;