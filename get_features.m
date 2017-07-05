function x = get_features(im, features, cell_size, cos_window)
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

temp = load('w2crs');
w2c = temp.w2crs;	
	if features.gray,
		%gray-level (scalar feature)
		x = double(im) / 255;
		
		x = x - mean(x(:));
    end
    
    if features.hsi, % Proposed Tracker
		%HSI Features, Convert to Reflectance and Add Noise
        im = NoiseAdd(im,0.9);
        % im = 200 * im;
        % Subtract from the Mean for Better Discrimination
        x = im - mean(mean(im));
        
        %HOG features, from Piotr's Toolbox
        im = sum(im(:,:,1:30),3);
		xHoG = double(fhog(single(im) / 255, cell_size, 9));
		xHoG(:,:,end) = [];  %remove all-zeros channel ("truncation feature")
        %out_pca = reshape(temp_pca, [prod(sz), size(temp_pca, 3)]);
		x = cat(3,x,xHoG);
        %x = xHoG;
        
    end
	
	%process with cosine window if needed
    
	if ~isempty(cos_window),
		x = bsxfun(@times, x, cos_window);
	end
	
end
