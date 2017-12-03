function x = conv_features(im, features, cell_size, cos_window, cnn_model)
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

%% Pre-Trained AlexNet - No Finetuning
im = NoiseAdd(im,0.1*randi([6 9])); %Flatten Spatially
img(:,:,1) = im(:,:,5); %Red
img(:,:,2) = im(:,:,15); %Red
img(:,:,3) = im(:,:,25); %Red
img = 255 * img ./ max(max(max(img))); %normalize the PCA comps
%Load Principal Components
img = preprocess_cnn(img);       %Preprocess the Image
res = cnn_model.forward({img});  %ForwardPass the Input Image
x = permute(res{1},[2 1 3]);
x = x / 1e3;


