%
%  Deep Hyperspectral Kernelized Correlation Filter Tracking
%
%  Burak Uzkent, 2017
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
%    Choose a FEATURE type, 'hog', 'gray' (raw pixels) or 'deep'.
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

% function run_tracker_hsi(kernel_type, feature_type,id) %Run it for specific target
function run_tracker_hsi(kernel_type, feature_type, grayImage) %Run it for all the targets

	%path to the videos (you'll be able to choose one with the GUI).
    base_path = '/Volumes/Burak_HardDrive/Moving_Platform_HSI_NoTrees_2/';
    
    %default settings
	if nargin < 1, kernel_type = 'gaussian'; end
	if nargin < 2, feature_type = 'hsi'; end

	%parameters according to the paper. at this point we can override
	%parameters based on the chosen kernel or feature type
	kernel.type = kernel_type;
	
	features.gray = false; %Initiate Feature Types
	features.hog = false;  % HoG
    features.deep_BN = false; % Binary Classifier - FineTuned AlexNet
	features.deep_HSI = false; % HSI Clasifier - FineTuned AlexNet
    features.deep_RGB = false; % Pre-Trained AlexNet
    features.hsi = false; % Raw HSI and HoG Features
    
	padding = 2.0;  %extra area surrounding the target
    cnnModel = [];
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
        interp_factor = 0.03; %Learning Rate
        kernel.sigma = 0.6; %Gaussian Bandwith

        kernel.poly_a = 1; %Polynomial Kernel
        kernel.poly_b = 9; %Polynomial Kernel

        features.hsi = true; %HSI Features
        cell_size = 1;
        
   case 'deep_HSI'  % Proposed Tracker Features
        interp_factor = 0.03; %Learning Rate
        kernel.sigma = 0.6; %Gaussian Kernel Bandwith

        kernel.poly_a = 1; %Polynomial Kernel
        kernel.poly_b = 9; %Polynomial Kernel
        features.deep_HSI = true; %Deep HSI Features
        cell_size = 1;
        %Deep CNN Model - AlexNet, GoogleNet Finetuned on Aerial Data Binary
        %Classification or Material Classification Network
        model = '/Volumes/Burak_HardDrive/Moving_Platform_CNN_Training/Caffe_Files_HSI/deploy.prototxt';
        weights = '/Volumes/Burak_HardDrive/Moving_Platform_CNN_Training/Caffe_Files_HSI/vd_net_iter_1000.caffemodel';
        cnnModel = caffe.Net(model, weights, 'test'); % create net and load weights

   case 'deep_BN'  % Proposed Tracker Features
        interp_factor = 0.03; %Learning Rate
        kernel.sigma = 0.6; %Gaussian Kernel Bandwith

        kernel.poly_a = 1; %Polynomial Kernel
        kernel.poly_b = 9; %Polynomial Kernel
        features.deep_BN = true; %Deep HSI Features
        cell_size = 1;
        %Deep CNN Model - AlexNet, GoogleNet Finetuned on Aerial Data Binary
        %Classification or Material Classification Network
        model = '/Volumes/Burak_HardDrive/Moving_Platform_CNN_Training/Caffe_Files/deploy.prototxt';
        weights = '/Volumes/Burak_HardDrive/Moving_Platform_CNN_Training/Caffe_Files/vd_net_iter_1000.caffemodel';     
        cnnModel = caffe.Net(model, weights, 'test'); % create net and load weights
        
    case 'deep_RGB'  % Proposed Tracker Features
        interp_factor = 0.03; %Learning Rate
        kernel.sigma = 0.6; %Gaussian Kernel Bandwith

        kernel.poly_a = 1; %Polynomial Kernel
        kernel.poly_b = 9; %Polynomial Kernel
        features.deep_RGB = true; %Deep HSI Features
        cell_size = 1;
        %Deep CNN Model - AlexNet
        %Classification or Material Classification Network
        %model = '/Users/buzkent/Documents/caffe/models/bvlc_reference_caffenet/deploy_Conv2.prototxt';
        model = '/Volumes/Burak_HardDrive/Moving_Platform_CNN_Training/VGG16/deploy_hsi.prototxt';
        %weights = '/Users/buzkent/Documents/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel';     
        weights = '/Volumes/Burak_HardDrive/Moving_Platform_CNN_Training/VGG16/VGG_ILSVRC_16_layers.caffemodel';     
        cnnModel = caffe.Net(model, weights, 'test'); % create net and load weights    
        
    % We will add more variants of the features from HSI
    otherwise
        error('Unknown feature.')
    end
    
	assert(any(strcmp(kernel_type, {'linear', 'polynomial', 'gaussian'})), 'Unknown kernel.')

    % -----------------------------------------------
    % Read the Ground Truth to Get Target Information - Read from Original
    % Text File to get Target Information
    file = dlmread('/Volumes/Burak_HardDrive/Moving_Platform_HSI/Ground_Truth/Vehicles_of_Interest.txt');
    counter = 1;
    for i = 1:size(file,1)
        id = i;
        target.id = file(id,1);
        target.firstFrame = file(id,2)+1;
        target.lastFrame = file(id,3);
        target.x = file(id,4) - 250; % For W/o Tree Scenario
        target.y = file(id,5); 
        target.width = 16;
        target.height = 16;
        target_sz = [target.width target.height];
        % ----------------------------------------------

        % Read the Homography Matrix
        target.H = [0.9994   -0.0107    8.1164;
            0.0069    0.9996   -5.1500;
            -0.0000   -0.0000    1.0000];

        % Call the tracker function with all the relevant parameters
        try
            pr_curve(counter,:) = tracker(base_path, target, target_sz, ...
                padding, kernel, lambda, output_sigma_factor, interp_factor, ...
                cell_size, features, cnnModel);
            counter
            pr_curve(counter,:)
            counter = counter+1;
        catch err
            continue;
        end 
    end
    %Plot the Figure
    close all
    pr = mean(pr_curve,1);
    plot(pr(1:50),'Linewidth',4);
    axis([0 50 0 1]);
    xlabel('Distance');
    ylabel('Precision');
    pr(51);
end
