function x = get_features(im, features, cell_size, cos_window, cnn_model)
%GET_FEATURES
%   Extracts dense features from image.
%
%   X = GET_FEATURES(IM, FEATURES, CELL_SIZE)
%   Extracts features specified in struct FEATURES, from image IM. The
%   features should be densely sampled, in cells or intervals of CELL_SIZE.
%   The output has size [height in cells, width in cells, features].
%
%   To specify HOG features, set field 'hog' to true, and
%   'hog_orientations' to the number of bins.
%
%   To experiment with other features simply add them to this function
%   and include any needed parameters in the FEATURES struct. To allow
%   combinations of features, stack them with x = cat(3, x, new_feat).
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/

% temp = load('w2crs');
% w2c = temp.w2crs;	
	if features.gray,
		%gray-level (scalar feature)
		x = double(im) / 255;
		
		x = x - mean(x(:));
    end
    
    if features.hsi, % Proposed Tracker
		%HSI Features, Convert to Reflectance and Add Noise
        im = NoiseAdd(im,0.1*randi([6 9])); %Flatten Spatially
        % Subtract from the Mean for Better Discrimination
        %x = im - mean(mean(im));
        %HOG features, from Piotr's Toolbox
        % im = sum(im(:,:,1:30),3);
        im = im(:,:,15);
		xHoG = double(fhog(single(im) / 255, cell_size, 9));
		xHoG(:,:,end) = [];  %remove all-zeros channel ("truncation feature")
        % out_pca = reshape(temp_pca, [prod(sz), size(temp_pca, 3)]);
        %x = cat(3,x,xHoG);
        x = xHoG;
        x = double(im_resize(x,[size(cos_window,1),size(cos_window,2)]));
    end
    % ----------------------------------------------
    if features.deep_BN, %Proposed Deep HKCF Tracker
        %% AlexNet FineTuned on Vehicle Detection on DIRSIG Dataset -------
        %HSI Features, Convert to Reflectance and Add Noise
        im = NoiseAdd(im,0.1*randi([6 9])); % Add Noise into the Image
        im = im(:,:,15);              %Randomly Pick a Band
        im = 255 * im / max(max(im));%Normalize the Band
        im = cat(3,im,im,im);         %GrayScale to RGB
        im = preprocess_cnn(im);      %Preprocess the Image
        res = cnn_model.forward({im}); %ForwardPass the Input Image
        x = permute(res{1},[2 1 3]);
        x = x / 1e3;
        x = double(im_resize(x,[size(cos_window,1),size(cos_window,2)],'bilinear'));
    end
    if features.deep_HSI
        %% Pretrained  AlexNet FineTuned on Vehicle-Material Detection on DIRSIG Dataset
        %Apply the PCA to the HSI Image
        load('/Volumes/Burak_HardDrive/envi/pca_coeff.mat');
        rows = size(im,1);
        cols = size(im,2);
        temp = reshape(NoiseAdd(im,0.1*randi([6 9])),rows*cols,61); %Flatten Spatially
        img(:,:,1) = reshape(sum(coeff(:,1)' .* temp,2),rows,cols,1); %Project to first
        img(:,:,2) = reshape(sum(coeff(:,2)' .* temp,2),rows,cols,1); %Project to Second
        img(:,:,3) = reshape(sum(coeff(:,3)' .* temp,2),rows,cols,1); %Project to Third
        img = 255 * img ./ max(max(max(img))); %normalize the PCA comps
        %Load Principal Components
        im = preprocess_cnn(img);       %Preprocess the Image
        res = cnn_model.forward({im});  %ForwardPass the Input Image
        %x = cnn_model.blobs('conv1').get_data(); %First Layer Conv Features
        x = permute(res{1},[2 1 3]);
        x = x / 1e3;
        x = double(im_resize(x,[size(cos_window,1),size(cos_window,2)],'bilinear'));
    end
    if features.deep_RGB
        %% Pre-Trained AlexNet - No Finetuning
        im = NoiseAdd(im,0.1*randi([6 9])); %Flatten Spatially
        img(:,:,1) = im(:,:,5); %Red
        img(:,:,2) = im(:,:,15); %Red
        img(:,:,3) = im(:,:,25); %Red
        img = 255 * img ./ max(max(max(img))); %normalize the PCA comps
        %Load Principal Components
        img = preprocess_cnn(img);       %Preprocess the Image
        res = cnn_model.forward({img});  %ForwardPass the Input Image
        %x = cnn_model.blobs('conv1').get_data(); %First Layer Conv Features
        x = permute(res{1},[2 1 3]);
        x = x / 1e3;
        x = double(im_resize(x,[size(cos_window,1),size(cos_window,2)]));
        %------------------------------------------------------------------
    end
    %process with cosine window if needed   
	if ~isempty(cos_window),
		x = bsxfun(@times, x, cos_window);
	end
	
end
