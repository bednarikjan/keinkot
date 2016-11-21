close all

%hyperIm = imread('../ortho.tif');
load('ortho_401x600.mat')
hyperIm = scaledIm;
%hyperIm = hyperIm(200:300, 300:470, :);
hyperIm = hyperIm(:,4:end,:);
% First channel in monochromatic in the 470-650 nm range
panChannel = hyperIm(:,:,1);
% Image coming from the VIS camera 470-650 nm
visIm = hyperIm(:,:,2:17);
% Image coming from the NIR camera 650-950 nm
nirIm = hyperIm(:,:,18:42);
% alpha channel of the image
alpha = hyperIm(:,:,43);
%% Regroup VIS and NIR images into reflectance structure, convert alpha to logical
refl = single(cat(3, visIm, nirIm));
alpha = alpha > 0;
%hyper.refl = bsxfun(@rdivide, hyper.refl, max(max(hyper.refl)));
% Scale integer data to 0..1 reflectance range
refl = (refl - min(refl(:))) / (max(refl(:)) - min(refl(:)));

%% Compute RGB, extract Refl size
rgb_corrected = refl(:,:,[16 8 2]);
rgb_corrected(:) = imadjust(rgb_corrected(:),stretchlim(rgb_corrected(:),[.01 .99]));
[n, m, p] = size(refl);

%% Preprocessing
%INCLUDE FOR TEST CASE 
% clear('rgb_corrected') %REMOVE!!!!!!!!!!!!!
% clear('refl')
% n = 10
% m = 20
% refl = zeros(n,m,p); %END REMOIVE

X = zeros(n*m,p);

for i=1:n
    for j=1:m
  %INCLUDE FOR TEST CASE
%         if i < 5 && j < 5
%             X(j+(i-1)*m,:) = ones(p,1);
%             refl(i,j,:)=ones(p,1);
%         elseif i < 7 && j < 7
%             X(j+(i-1)*m,:) = [ones(floor(p/2),1); zeros(p-floor(p/2),1)]';
%             refl(i,j,:)=[ones(floor(p/2),1); zeros(p-floor(p/2),1)]';
%         end

%EXCLUDE FOR TEST CASE
        X(j+(i-1)*m,:) = refl(i,j,:);
    end
end

%INCLUDE for test case
%rgb_corrected = refl(:,:,[16 8 2]);
%nirIm = refl(:,:,17:41);
 
% %remove lowest frequencies - only works for odd length of data
% %does not seem to improve at the moment
% assert(mod(length(size(X,2)), 2)==1); 
% mask = ones(1,size(X,2));
% mask(1:5) = 0; mask(end-4:end) = 0; %without second part the ifft is
% complex 
% mask = repmat(mask, size(X, 1), 1);
% %is ifftshift actually different form fftshift?
% X = ifft(ifftshift(fftshift(fft(X')).*mask'))';

%Xstd = zscore(X')';
%Xstd = (X - min(X,[],2)*ones(1,p)) ./ ((max(X,[],2) - min(X,[],2))*ones(1,p));
Xstd = X ./ (repmat(sqrt(sum(X.^2, 2)),1,p) + eps);

%% Do Principal component analysis

[features,score,latent,tsquare] = princomp(zscore(X));
[maximums,classes] = max(score,[],2);
fig=figure
plot(latent,'*');
savefig(fig,'results/princomp.fig')


%% K means classification
while 1  %until the final cluster is not empty  
    K = 5;
    [classes, features] = kmeans(Xstd,K,'Display','iter');
    %assert(K == size(features,1));
    if norm(features(K, :)) > 0.8 %full clusters will have a norm close to 1, empty close to 0
        break
    end
end

fig=figure
for class=1:K
    plot(1:size(features,2), features(class,:),'DisplayName',num2str(class))
    hold on
end
legend('show')

[a,b] = hist(classes, unique(classes));

classes_img = zeros(n,m,3);
colors = hsv(p);
%%
fig=figure
classes = reshape(classes, m, n)';
classes(isnan(classes)) = 6;
img_labels = label2rgb(classes);

%
for class=1:n
    for j=1:m
        classes_img(class,j,:) = 255.0/10.0*classes(j+(class-1)*m);
        %classes_img(i,j,:) = colors(classes(j+i*m));
        
    end
end


%% Smoothing the cluster asignment.

%%% Morphological operations - do not work as expected -> do not use them

% nbghood = true(4);
% 
% % Smooth each cluster separately.
% for i = 1:K
%     % Substituting all the labels not corresponding to this class to 0
%     binClust = classes;
%     binClust(binClust ~= i) = 0;
%     % Making this binary (I in {0, 1}^WxH)
%     label = max(max(binClust));
%     binClust = binClust / label;
%     
%     % Smooth using dilation followed by erosion.
%     smoothBinClust = imclose(binClust, nbghood);
%     smoothBinClust = smoothBinClust * label;
%     binClusters(:, :, i) = smoothBinClust;
% end
% 
% % Merge the clusters again in one 'classes' image
% smoothedClasses = zeros(size(classes));
% for i = 1:K
%     smoothedClasses(binClusters(:, :, i) == i) = i;
% end
% 
% img_lables_smoothed = label2rgb(smoothedClasses');
% 
% image(2)
% subplot(211)
% imshow(img_labels)
% subplot(212)
% imshow(img_lables_smoothed)

%%%  Smoothing using median filter %%%
smoothedClassesMed = medfilt2(classes, [5 5]);

img_lables_smoothed_med = label2rgb(smoothedClassesMed);
image(2)
subplot(211)
imshow(img_labels)
subplot(212)
imshow(img_lables_smoothed_med)
savefig(fig,'results/clustering.fig')


% Rewriting the original classes with smoothed ones.
classes = smoothedClassesMed;

%% Binary classification of each class
%close all
nir_imread = uint8(mean(nirIm, 3));

[shadow, thres] = compute_shadow(uint8(rgb_corrected), nir_imread );
threshold = 0.83;
maybe_cloud = shadow < threshold;
fig=figure
subplot 211
imshow(maybe_cloud)
title('global sabine shadowmask');
subplot 212
imshow(rgb_corrected)
title('rgb image');


%fig=figure
N_nc = 500; % number of non-cloud pixels
N = m*n;
M = 1; % number of frames
g = ones(p,1);
Xcorrected = refl;%zeros(n,m,p);
shadowmap = zeros(n,m);
correctedmap = zeros(n,m);

%options = optimset('Display', 'off') ;
options = optimset('Display','on');
for class=1:K
    corrected_points = zeros(n,m);
    class_pixels = (classes == class);
    %subplot(1,K,i)
    mask_maybe_cloud = (class_pixels.*maybe_cloud)==1;
    pixels_maybe_cloud = X(mask_maybe_cloud(:),:);
    mask_no_cloud = (class_pixels.*(1-maybe_cloud))==1;
    pixels_no_cloud = X(mask_no_cloud(:),:);
    row_index_of_X = repmat((1:n)', 1, m);
    column_index_of_X = repmat(1:m, n, 1);
    indices_row_maybe_cloud = row_index_of_X(mask_maybe_cloud);
    indices_column_maybe_cloud = column_index_of_X(mask_maybe_cloud);
    
    %assert(size(pixels_maybe_cloud,1) + size(pixels_no_cloud, 1)==n*m)
    
    % choose N_nc non clouded points 
    [B, cols] = sort(sum(pixels_no_cloud.^2,2));
    disp('chosen non-clouded points:')
    cols(end-N_nc-10:end-10)
    I_nc = X(cols(1:N_nc),:);
    
    d0 = 0.2;
    d = zeros(N_nc);
    costs = zeros(1,N_nc);

    for i = 1:n %N_maybe
        for j = 1:m
        I_i = X(j+(i-1)*m,:); 
        thicknesses = 1;
        I_maybe_cloud_corrected =  I_i ./thicknesses;
        Xcorrected(i, j,:) = I_maybe_cloud_corrected;         
   %     norm(squeeze(Xcorrected(indices_row_maybe_cloud(i), indices_column_maybe_cloud(i),:))' - I_i)
        %assert(squeeze(Xcorrected(indices_row_maybe_cloud(i), indices_column_maybe_cloud(i),:)) == I_i)
        %assert(Xcorrected(indices_row_maybe_cloud(i), indices_column_maybe_cloud(i),:) == I_i); 
        %assert(X(indices_row_maybe_cloud(i), indices_column_maybe_cloud(i),:) == I_i); 
        corrected_points(i,j) = 1;
        shadowmap(i,j) = max(thicknesses);
        correctedmap = correctedmap + corrected_points;
        end
    end
    
%     for i = 1:sum(sum(mask_maybe_cloud)) %N_maybe
%         I_i = X(i,:); 
%         for non_cloud_pixel=1:N_nc
%             I_j = I_nc(non_cloud_pixel,:);            
%             %cost = sum(sum((I_nc_tilde - I_nc_j).^2))  
%         %    fun = @(d)objective(d,g,I_i,I_j);
%         %    %test = fmincon(fun, d0, [],[],[],[],0,Inf); 
%         %    [d(non_cloud_pixel,:),costs(non_cloud_pixel)] = fmincon(fun, d0, [],[],[],[],zeros(M,1), ones(M,1),[],options); 
%             %d(non_cloud_pixel) = I_j*I_i'/(I_j*I_j');
%             %costs(non_cloud_pixel) = norm(I_i - I_j*d(non_cloud_pixel))^2;
%           
%         end
%         %thicknesses = d(unique(costs==min(costs)));
%         thicknesses = 1;
%   %      I_maybe_cloud_corrected =  I_i .* sum(exp(thicknesses'*g'),1)/M;
%         I_maybe_cloud_corrected =  I_i ./thicknesses;
%         Xcorrected(indices_row_maybe_cloud(i), indices_column_maybe_cloud(i),:) = I_maybe_cloud_corrected;         
%    %     norm(squeeze(Xcorrected(indices_row_maybe_cloud(i), indices_column_maybe_cloud(i),:))' - I_i)
%         %assert(squeeze(Xcorrected(indices_row_maybe_cloud(i), indices_column_maybe_cloud(i),:)) == I_i)
%         %assert(Xcorrected(indices_row_maybe_cloud(i), indices_column_maybe_cloud(i),:) == I_i); 
%         %assert(X(indices_row_maybe_cloud(i), indices_column_maybe_cloud(i),:) == I_i); 
%         corrected_points(indices_row_maybe_cloud(i), indices_column_maybe_cloud(i)) = 1;
%         shadowmap(indices_row_maybe_cloud(i), indices_column_maybe_cloud(i)) = max(thicknesses);
%         correctedmap = correctedmap + corrected_points;
%     end
    fig=figure
    subplot 211
    imshow(corrected_points)
    title(['we have touched these points because they are in maybe cloud of ',num2str(class)])
    subplot 212
    imshow(mask_maybe_cloud)
    title(['mask maybe cloud of class ',num2str(class)])
    savefig(fig,['results/class',num2str(class),'.fig'])
end
%%

% 
fig=figure
imshow(shadowmap)
title('global shadowmap (inferred)')
savefig(fig,'results/shadowmap.fig')

fig=figure
subplot 311
imshow(rgb_corrected)
title('rgb image');

subplot 312
imshow(correctedmap)
title('all points corrected')

% fig=figure
% imshow(corrected_points)
subplot 313
rgb_corrected_final = Xcorrected(:,:,[16 8 2]);
rgb_corrected_final(:) = imadjust(rgb_corrected_final(:),stretchlim(rgb_corrected_final(:),[.01 .99]));
imshow(rgb_corrected_final)
title('final result')
savefig(fig,'results/summary.fig')

    %I_nc_tilde = 
    % Create histogram of this class
  %  size(X((classes==i),:))
  %  size(sum(X((classes==i),:),2))
    %hist(sum(X((classes==i),:),2),p)
    %title(['Class ',num2str(i)])