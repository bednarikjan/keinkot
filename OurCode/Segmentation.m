close all

%hyperIm = imread('../ortho.tif');
load('ortho_401x600.mat')
hyperIm = scaledIm;
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
rgb = refl(:,:,[16 8 2]);
rgb(:) = imadjust(rgb(:),stretchlim(rgb(:),[.01 .99]));
[n, m, p] = size(refl);

%% Preprocessing
X = zeros(n*m,p);
for class=1:n
    for j=1:m
        X(j+(class-1)*m,:) = refl(class,j,:);
    end
end
 
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
figure
plot(latent,'*')



%% K means classification
while 1  %until the final cluster is not empty  
    K = 5;
    [classes, features] = kmeans(Xstd,K,'Display','iter');
    %assert(K == size(features,1));
    if norm(features(K, :)) > 0.8 %full clusters will have a norm close to 1, empty close to 0
        break
    end
end

figure
for class=1:K
    plot(1:size(features,2), features(class,:),'DisplayName',num2str(class))
    hold on
end
legend('show')

[a,b] = hist(classes, unique(classes));

classes_img = zeros(n,m,3);
colors = hsv(p);
%%
figure
classes = reshape(classes, m, n);
classes(isnan(classes)) = 6;
img_labels = label2rgb(classes');

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

img_lables_smoothed_med = label2rgb(smoothedClassesMed');

image(2)
subplot(211)
imshow(img_labels)
subplot(212)
imshow(img_lables_smoothed_med)

figure()
image(rgb)

% Rewriting the original classes with smoothed ones.
classes = smoothedClassesMed

%% Binary classification of each class
close all
nir_imread = uint8(mean(nirIm, 3));

[shadow, thres] = compute_shadow(uint8(rgb), nir_imread );
figure
threshold = 0.83;
mask = shadow < threshold;
figure
subplot 211
imshow(mask)
subplot 212
imshow(rgb)


I_obs = X

%figure
N_nc = 3; % number of non-cloud pixels
N = m*n;
M = 10; % number of frames
g = ones(p,1);

for class=1:1%K
    pixels = (classes == class);
    %subplot(1,K,i)
    indices_in_cloud = ((classes==class)'.*mask)==1;
    pixels_in_cloud = X(indices_in_cloud,:);
    indices_no_cloud = ((classes==class)'.*mask)==0;
    pixels_no_cloud = X(indices_no_cloud,:);
    assert(size(pixels_in_cloud,1) + size(pixels_no_cloud, 1)==n*m)
    
    % choose N_nc non clouded points 
    [B, cols] = sort(shadow(classes == class));
    I_nc = X(cols(1:N_nc),:);
    
    d = zeros(M,1);
    for non_cloud_pixel=1:N_nc
        for pixel = 1:N           
            I_nc_tilde = M * I_obs(pixel,:) ./ sum(exp(repmat(d,N).*repmat(g,M)));
            I_nc_j = I_nc(non_cloud_pixel,:);
            cost = sum(sum((I_nc_tilde - I_nc_j).^2));
            print(cost)
        end
    end
end
    %I_nc_tilde = 
    % Create histogram of this class
  %  size(X((classes==i),:))
  %  size(sum(X((classes==i),:),2))
    %hist(sum(X((classes==i),:),2),p)
    %title(['Class ',num2str(i)])