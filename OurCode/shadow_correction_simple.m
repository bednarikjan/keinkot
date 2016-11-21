% Load the image.
load('ortho_401x600.mat')
hyperIm = scaledIm;

% Load class labels.
load('classes_5_401x600.mat')

% Get some shit needed for shadow mask computation.
visIm = hyperIm(:,:,2:17);
nirIm = hyperIm(:,:,18:42);
alpha = hyperIm(:,:,43);
% Reflectance - the image which we actually work with, 41 channels.
refl = single(cat(3, visIm, nirIm));
refl = (refl - min(refl(:))) / (max(refl(:)) - min(refl(:)));
nirIm = hyperIm(:,:,18:42);
nir_imread = uint8(mean(nirIm, 3));
rgb = refl(:,:,[16 8 2]);
rgb(:) = imadjust(rgb(:),stretchlim(rgb(:),[.01 .99]));

% Get shadow mask.
[shadow, thres] = compute_shadow(uint8(rgb), nir_imread );

% Finally get this thing done.
% clean_clouds(refl, shadow, classLabels)
cleanedImage = clean_clouds_simple(refl, shadow, classLabels);

rgbCleaned = cleanedImage(:, :, [16 8 2]);

imshow(rgbCleaned)
