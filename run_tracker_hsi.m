%
%  High-Speed Tracking with Kernelized Correlation Filters
%
%  Joao F. Henriques, 2014
%  http://www.isr.uc.pt/~henriques/
%
%  Main interface for Kernelized/Dual Correlation Filters (KCF/DCF).
%  This function takes care of setting up parameters, loading video
%  information and computing precisions. For the actual tracking code,
%  check out the TRACKER function.
%
%  RUN_TRACKER
%    Without any parameters, will ask you to choose a video, track using
%    the Gaussian KCF on HOG, and show the results in an interactive
%    figure. Press 'Esc' to stop the tracker early. You can navigate the
%    video using the scrollbar at the bottom.
%
%  RUN_TRACKER VIDEO
%    Allows you to select a VIDEO by its name. 'all' will run all videos
%    and show average statistics. 'choose' will select one interactively.
%
%  RUN_TRACKER VIDEO KERNEL
%    Choose a KERNEL. 'gaussian'/'polynomial' to run KCF, 'linear' for DCF.
%
%  RUN_TRACKER VIDEO KERNEL FEATURE
%    Choose a FEATURE type, either 'hog' or 'gray' (raw pixels).
%
%  RUN_TRACKER(VIDEO, KERNEL, FEATURE, SHOW_VISUALIZATION, SHOW_PLOTS)
%    Decide whether to show the scrollable figure, and the precision plot.
%
%  Useful combinations:
%  >> run_tracker choose gaussian hog  %Kernelized Correlation Filter (KCF)
%  >> run_tracker choose linear hog    %Dual Correlation Filter (DCF)
%  >> run_tracker choose gaussian gray %Single-channel KCF (ECCV'12 paper)
%  >> run_tracker choose linear gray   %MOSSE filter (single channel)
%

function run_tracker_hsi(kernel_type, feature_type,id)

	%path to the videos (you'll be able to choose one with the GUI).
    base_path = '/Volumes/Seagate Backup Plus Drive/Moving_Platform_HSI/';

	%default settings
	if nargin < 1, kernel_type = 'gaussian'; end
	if nargin < 2, feature_type = 'hsi'; end

	%parameters according to the paper. at this point we can override
	%parameters based on the chosen kernel or feature type
	kernel.type = kernel_type;
	
	features.gray = false;
	features.hog = false;
	
	padding = 2.0;  %extra area surrounding the target

	lambda = 1e-4;  %regularization
	output_sigma_factor = 0.10;  %spatial bandwidth (proportional to target)
	
	switch feature_type
    case 'gray',
        interp_factor = 0.075;  %linear interpolation factor for adaptation

        kernel.sigma = 0.2;  %gaussian kernel bandwidth

        kernel.poly_a = 1;  %polynomial kernel additive term
        kernel.poly_b = 7;  %polynomial kernel exponent

        features.gray = true;
        cell_size = 1;

   case 'hsi'  % Proposed Tracker Features
        interp_factor = 0.03;
        kernel.sigma = 0.6;

        kernel.poly_a = 1;
        kernel.poly_b = 9;

        features.hsi = true;
        cell_size = 1;

    % We will add more variants of the features from HSI

    otherwise
        error('Unknown feature.')
    end
    
	assert(any(strcmp(kernel_type, {'linear', 'polynomial', 'gaussian'})), 'Unknown kernel.')

    % -----------------------------------------------
    % Read the Ground Truth to Get Target Information - Read from Original
    % Text File to get Target Information
    file = dlmread('/Volumes/Seagate Backup Plus Drive/Moving_Platform_HSI/Ground_Truth/Vehicles_of_Interest.txt');
    target.id = file(id,1);
    target.firstFrame = file(id,2)+1;
    target.lastFrame = file(id,3);
    target.x = file(id,4);
    target.y = file(id,5);
    target.width = file(id,6)*2;
    target.height = file(id,7)*2;
    target_sz = [target.width target.height];
    % ----------------------------------------------
    
    % Read the Homography Matrix
    target.H = [0.9994   -0.0107    8.1164;
        0.0069    0.9996   -5.1500;
        -0.0000   -0.0000    1.0000];
    
    % Call the tracker function with all the relevant parameters
    tracker(base_path, target, target_sz, ...
        padding, kernel, lambda, output_sigma_factor, interp_factor, ...
        cell_size, features);
		
end
