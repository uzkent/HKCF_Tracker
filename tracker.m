function [pr_curve] = tracker(base_path, target, target_sz, ...
	padding, kernel, lambda, output_sigma_factor, interp_factor, cell_size, ...
	features)
%TRACKER Kernelized/Dual Correlation Filter (KCF/DCF) tracking.
%   This function implements the pipeline for tracking with the KCF (by
%   choosing a non-linear kernel) and DCF (by choosing a linear kernel).
%
%   It is meant to be called by the interface function RUN_TRACKER, which
%   sets up the parameters and loads the video information.
%
%   Parameters:
%     VIDEO_PATH is the location of the image files (must end with a slash
%      '/' or '\').
%     IMG_FILES is a cell array of image file names.
%     POS and TARGET_SZ are the initial position and size of the target
%      (both in format [rows, columns]).
%     PADDING is the additional tracked region, for context, relative to 
%      the target size.
%     KERNEL is a struct describing the kernel. The field TYPE must be one
%      of 'gaussian', 'polynomial' or 'linear'. The optional fields SIGMA,
%      POLY_A and POLY_B are the parameters for the Gaussian and Polynomial
%      kernels.
%     OUTPUT_SIGMA_FACTOR is the spatial bandwidth of the regression
%      target, relative to the target size.
%     INTERP_FACTOR is the adaptation rate of the tracker.
%     CELL_SIZE is the number of pixels per cell (must be 1 if using raw
%      pixels).
%     FEATURES is a struct describing the used features (see GET_FEATURES).
%     SHOW_VISUALIZATION will show an interactive video if set to true.
%
%   Outputs:
%    POSITIONS is an Nx2 matrix of target positions over time (in the
%     format [rows, columns]).
%    TIME is the tracker execution time, without video loading/rendering.
%
%   Joao F. Henriques, 2014

%   window size, taking padding into account
	window_sz = floor(target_sz * (1 + padding));
	% window_sz = [100 100]; % For now we use a fixed ROI
    
% 	we could choose a size that is a power of two, for better FFT
% 	%performance. in practice it is slower, due to the larger window size.
% 	window_sz = 2 .^ nextpow2(window_sz);
	
	%create regression labels, gaussian shaped, with a bandwidth
	%proportional to target size
	output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
    if isfield(features, 'deep') && features.deep
        yf = fft2(gaussian_shaped_labels(output_sigma, ceil(window_sz / cell_size)));
    else
        yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));
    end

	%store pre-computed cosine window - to avoid distortion due to FFT
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';	
	
	%note: variables ending with 'f' are in the Fourier domain.

	time = 0;  %to calculate FPS
    
    frameCounter = 1;
	for frame = target.firstFrame:target.lastFrame,
		%load HSI Image - Handle - For now keep reading the same imagOOooe
        imgHandle = matfile([base_path 'Image_' num2str(frame) '.mat']);
        
        %Load GrayScale Imagery for Display Purposes
        % grayImage = imgHandle.img(:,:,1);
        
		tic();
        if frameCounter > 1
            
            %Apply Homograpy to Previous Position
            applyHomograpy(target,frameCounter-1);
            
            %Sample The ROI From the Full Image
            xCoord = min(max(1,target.x-(window_sz(1)/2)*2),1500):min(max(1,target.x+(window_sz(1)/2)*2),1500);
            yCoord = min(max(1,target.y-(window_sz(2)/2)*2),1500):min(max(1,target.y+(window_sz(2)/2)*2),1500);
            hsi_roi = imgHandle.img(xCoord,yCoord,:);
            
            SubWindowsX = round(linspace(1,size(xCoord,2)-window_sz(1),5));
            SubWindowsY = round(linspace(1,size(yCoord,2)-window_sz(2),5));
            for i = 1:5
                for j = 1:5
            
                    %obtain a subwindow for detection at the position from last
                    xSubWindow = SubWindowsX(i):SubWindowsX(i)+window_sz(1)-1;
                    ySubWindow = SubWindowsY(j):SubWindowsY(j)+window_sz(2)-1;
                    SubWindowX{i,j} = xCoord(1) + xSubWindow(end/2); 
                    SubWindowY{i,j} = yCoord(1) + ySubWindow(end/2); 
                    roi = hsi_roi(xSubWindow,ySubWindow,:);
                    
                    %frame, and convert to Fourier domain (its size is unchanged)
                    zf = fft2(get_features(roi, features, cell_size, cos_window));

                    %calculate response of the classifier at all shifts
                    switch kernel.type
                    case 'gaussian',
                        kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
                    case 'polynomial',
                        kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
                    case 'linear',
                        kzf = linear_correlation(zf, model_xf);
                    end
                    response = real(ifft2(model_alphaf .* kzf));  %equation for fast detection
            
                    %target location is at the maximum response. we must take into
                    %account the fact that, if the target doesn't move, the peak
                    %will appear at the top-left corner, not at the center (this is
                    %discussed in the paper). the responses wrap around cyclically.
                    [vert(i,j), horiz(i,j)] = find(response == max(response(:)), 1);
                    confidence(i,j) = max(max(response)); %Confidence of Tracker
                    
                    %Store Subwindows for Debugging
                    dROI{i,j} = roi(:,:,1);
             
                end
            end
                %Shift the tracker to new position
                [iX,iY] = find(confidence == max(max(confidence)),1);
                vert_delta = vert(iX,iY);
                horiz_delta = horiz(iX,iY);
                if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
                    vert_delta = vert_delta - size(zf,1);
                end
                if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
                    horiz_delta = horiz_delta - size(zf,2);
                end
                target.x = SubWindowX{iX,iY} + cell_size * [vert_delta - 1];
                target.y = SubWindowY{iX,iY} + cell_size * [horiz_delta - 1];
           
        end
        
        %obtain a subwindow for training at newly estimated target position
        % Sample The ROI From the Full Image
        xCoord = min(max(1,target.x-(window_sz(1)/2)),1500):min(max(1,target.x+(window_sz(1)/2)),1500)-1;
        yCoord = min(max(1,target.y-(window_sz(2)/2)),1500):min(max(1,target.y+(window_sz(2)/2)),1500)-1;
        hsi_roi = imgHandle.img(xCoord,yCoord,:);
        xf = fft2(get_features(hsi_roi, features, cell_size, cos_window));

        %Kernel Ridge Regression, calculate alphas (in Fourier domain)
        switch kernel.type
        case 'gaussian',
            kf = gaussian_correlation(xf, xf, kernel.sigma);
        case 'polynomial',
            kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
        case 'linear',
            kf = linear_correlation(xf, xf);
        end
        alphaf = yf ./ (kf + lambda);   %equation for fast training
   
		if frameCounter == 1,  %first frame, train with a single image
			model_alphaf = alphaf;
			model_xf = xf;
		else
			%subsequent frames, interpolate model
			model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
			model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
        end

        %Transfer the results to global array
        time = toc();
        results(frameCounter,:) = [target.x target.y frame time];
        frameCounter = frameCounter + 1;
		% save position and timing
        
        figure(1);
        imshow(hsi_roi(:,:,1),[]);
        % hold on
        % plot(target.y,target.x,'go','Linewidth',3);
		
    end
    
    %Compute Precision
    pr_curve = precision_curve(target,results);

end

