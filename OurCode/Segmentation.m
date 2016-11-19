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
clear('visIm','nirIm');
%hyper.refl = bsxfun(@rdivide, hyper.refl, max(max(hyper.refl)));
% Scale integer data to 0..1 reflectance range
refl = (refl - min(refl(:))) / (max(refl(:)) - min(refl(:)));

%% Compute RGB, extract Refl size
rgb = refl(:,:,[16 8 2]);
rgb(:) = imadjust(rgb(:),stretchlim(rgb(:),[.01 .99]));
[n, m, p] = size(refl);

%% Preprocessing
X = zeros(n*m,p);
for i=1:n
    for j=1:m
        X(j+(i-1)*m,:) = refl(i,j,:);
    end
end

%Xstd = zscore(X')';
%Xstd = (X - min(X,[],2)*ones(1,p)) ./ ((max(X,[],2) - min(X,[],2))*ones(1,p));
Xstd = X ./ (repmat(sqrt(sum(X.^2, 2)),1,p) + eps);

%% Do Principal component analysis

[features,score,latent,tsquare] = princomp(zscore(X));
[maximums,classes] = max(score,[],2);
figure
plot(latent,'*')

K = 6;

%% K means classification

[classes, features] = kmeans(Xstd,K,'Display','iter');
assert(K == size(features,1));
figure
for i=1:K
    plot(1:size(features,2), features(i,:),'DisplayName',num2str(i))
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
for i=1:n
    for j=1:m
        classes_img(i,j,:) = 255.0/10.0*classes(j+(i-1)*m);
        %classes_img(i,j,:) = colors(classes(j+i*m));
    end
end

figure
%%
subplot 211
img_labels=imfilter(double(img_labels),fspecial('gaussian',5));
imshow(uint8(img_labels))
colorbar
subplot 212
image(rgb)
%cumsum(latent)./sum(latent);

%% Binary classification of each class




