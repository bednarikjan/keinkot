function [ cleanImage ] = clean_clouds_simple(image, shadowMaskSoft, classLabels )
%clean_clouds This function cleans the shadowy regions given the following:
% INPUTS:
% * image : (W x H x Ch)-tensor, single, 
%       The actual image, W - width, H - height, Ch - # channels
% * shadowMaskSoft : (W x H)-matrix, double
%       Soft shadow mask in range [0.0, 1.0], where 0.0 == maximum cloud, 1.0 == no cloud at all. 
% * classLabels : (W x H)-matrix, uint8
%       Class labels.
% 
% OUTPUTS:
% * cleanImage (over all the channels)
% The function computes the cloud effect over each channel and for multiple
%   ranges of mask values (over every class), then cancels out the cloud
%   efffect.

% Resulting image.
cleanImage = image;

% Make the class labels the same dimensions as input image.
% classLabels = classLabels';

% Get number of classes.
numClasses = max(max(classLabels));
disp(['Number of classes: ' num2str(numClasses)]);

% Get number of image channels.
numChannels = size(image, 3);
disp(['Number of channels: ' num2str(numChannels)]);

% Binary shadow mask (1 == no cloud), (0 == cloud)
% shadowThreshold = 0.84;
shadowThreshold = 0.85;
% Everything which is higher then this threshold is NOT a shadow.
notShadowMaskBin = shadowMaskSoft > shadowThreshold;

% Shadow masks over M sub-ranges
M = 60;
shadowThMax = shadowThreshold;
shadowThMin = 0.1;
shadowThresholds = linspace(shadowThMin, shadowThMax, M + 1)

for m = 1:M
    msk = logical((shadowMaskSoft >= shadowThresholds(m)) .* (shadowMaskSoft < shadowThresholds(m + 1)));
    notShadowMasks(:, :, m) = msk;
end

% debug
% imshow(shadowMaskBin)

% Process each class independently.
for clsLabel = 1:numClasses
    
    % Get mask corresponding to given class.
    classMask = (classLabels == clsLabel);
    
    % Mask for subset of pixels which are definitely NOT shadow at all.
    classNotShadowMask = logical(classMask .* notShadowMaskBin);
    
    % Process the image for each subrange of shadow intensity separately.
    for smi = 1:M
        % Mask for subset of pixels which ARE shadow.
%         classShadowMask = logical(classMask .* (1 - notShadowMaskBin));
        classShadowMask = logical(classMask .* notShadowMasks(:, :, smi));

        % Sanity checks about cloud and not cloud within class masks.
%         assert(sum(sum(classNotShadowMask)) + sum(sum(classShadowMask)) == sum(sum(classMask)));
%         assert(sum(sum(logical(classNotShadowMask .* classShadowMask))) == 0);

        % Average pixel in NOT shadow subset and average pixel in shadow subset.
        avgNotShadowPixel = zeros(numChannels, 1);
        avgShadowPixel = zeros(numChannels, 1);

        % Compute the average separately over each channel.
        for ch = 1:numChannels
           imageCh = image(:, :, ch); 
           avgNotShadowPixel(ch) = mean(mean(imageCh(classNotShadowMask)));
           avgShadowPixel(ch) = mean(mean(imageCh(classShadowMask)));
        end

        % Compute shift between not shadow and shadow pixel average.
        shift = avgNotShadowPixel ./ avgShadowPixel;

        % Shift all the shadow pixels by the shift.
        for ch = 1:numChannels
            imageCh = cleanImage(:, :, ch);
            imageCh(classShadowMask) = imageCh(classShadowMask) * shift(ch);
            cleanImage(:, :, ch) = imageCh;
        end
    end
end

% for chIdx = 1:numChannels
%     % debug
%     imshow(cleanImage(:, :, chIdx));
%     pause;
% end

