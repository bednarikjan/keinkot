function [ cleaned_image ] = clean_clouds( I, shadow_mask, class_ids )
%clean_clouds this function cleans the shadowy regions given the following:
% INPUTS:
% * Segmented map into different classes
% * Cloud shadow mask 
% * Actual image (with all its 43 channels)
% OUTPUTS:
% * cleaned_image (over all the channels)
% The function computes the cloud effect over each channel and for multiple
%   ranges of mask values (over every class), then cancels out the cloud
%   efffect

class_ids = class_ids';

%% Find shadowy points 
thresh_mask_S1 = 0.825;
thresh_mask_S2 = 0.8;
thresh_mask_S3 = 0.775;
thresh_mask_NS = 0.825;
mask_shadowy_1   = (shadow_mask < thresh_mask_S1) .* (shadow_mask >= thresh_mask_S2);
mask_shadowy_2   = (shadow_mask < thresh_mask_S2) .* (shadow_mask >= thresh_mask_S3);
mask_shadowy_3   = shadow_mask < thresh_mask_S3;
mask_not_shadowy = shadow_mask > thresh_mask_NS;

%% For each class: pick a given pixel, fload around it, work on that geometric 
% cluster, mark that this component cluster is done, then pick another
% pixel in this class, fload and repeat till no more such pixels exist,
% then move to the next class

nbr_classes = max(max( class_ids ));
cleaned_image = I;

for class = 1:nbr_classes
    %find all little components:
    class_mask = class_ids == class;
    covered_all_clusters = 0;
    while ~covered_all_clusters
        [point_x, point_y] = find(class_mask == 1);
        random_point_x = point_x(1);
        random_point_y = point_y(1);
        %flood around this random coordinate couple
        flooded_comp_mask = grayconnected(uint8(class_mask), random_point_x, random_point_y, 0.1);
        
        flood_and_shadow_1 = flooded_comp_mask .* mask_shadowy_1;
        flood_and_shadow_2 = flooded_comp_mask .* mask_shadowy_2;
        flood_and_shadow_3 = flooded_comp_mask .* mask_shadowy_3;
        
        flood_and_notshadow = uint8(flooded_comp_mask .* mask_not_shadowy);
        
        for w = 1:size(I,3)
            masked_I_NS = I(:,:,w) .* flood_and_notshadow;
            try
            mean_intensities_NS(w) = sum(sum(masked_I)) / sum(sum(flood_and_notshadow));
            catch
                disp(['No clean entries in this patch, its size is:' num2str(sum(sum(flooded_comp_mask))) ]);
            end
            
            try
            masked_I_S_1 = I(:,:,w) .* flood_and_shadow_1;
            mean_intensities_S_1(w) = sum(sum(masked_I_S_1)) / sum(sum(flood_and_shadow_1));
            RATIO_1(w) = mean_intensities_S_1(w) / mean_intensities_NS_1(w);
            cleaned_image( masked_I_S_1 == 1, w ) = cleaned_image( masked_I_S_1 == 1, w ) * RATIO_1(w);
            catch
            end

            try
            masked_I_S_2 = I(:,:,w) .* flood_and_shadow_2;
            mean_intensities_S_2(w) = sum(sum(masked_I_S_2)) / sum(sum(flood_and_shadow_2));
            RATIO_2(w) = mean_intensities_S_2(w) / mean_intensities_NS_2(w);
            cleaned_image( masked_I_S_2 == 1, w ) = cleaned_image( masked_I_S_2 == 1, w ) * RATIO_2(w);
            catch
            end

            try
            masked_I_S_3 = I(:,:,w) .* flood_and_shadow_3;
            mean_intensities_S_3(w) = sum(sum(masked_I_S_3)) / sum(sum(flood_and_shadow_3));
            RATIO_3(w) = mean_intensities_S_3(w) / mean_intensities_NS_3(w);
            cleaned_image( masked_I_S_3 == 1, w ) = cleaned_image( masked_I_S_3 == 1, w ) * RATIO_3(w);
            catch
            end
        end 
        
        class_mask = class_mask - flooded_comp_mask;
        if( sum(sum(class_mask)) < 15 )
            covered_all_clusters = 1;
        end

    end
  
end