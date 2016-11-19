function [ shadow, thres ] = compute_shadow( vis_imread, nir_imread )
%compute_shadow returns both the shadow map and a good thres
%  the inputs are your outputs of imread() AS IS!

    %Apply our method for all images
    disp('Computing shadow mask for our method...')
    a = 14.0; %Determines the slope of the sigmoid function. Default: 14.0
    b = 0.5; %Inflection point of sigmoid. Default: 0.5
    thresh = 10.0; %thresh is the threshold used in the color to NIR computation. Default: 10.0
    useratios = 1; %Set this to zero if you don't want to use color to NIR ratios.
    gamma = 2.2; %To stretch the shadows in the tone mapping function before applying the sigmoid function f
    
    %Compute the shadow masks for all images that are is the "images_folder"
%         disp(['Processing image ' filenames_vis(i).name ' (' num2str(i) '/' num2str(length(filenames_vis))  ')...']);
        
        vis = im2double(vis_imread);
        try
            nir = rgb2gray(im2double(nir_imread));
        catch
            nir = im2double(nir_imread);
        end
        m1=min(min(vis)); M1=max(max(vis));
        m2=min(min(nir)); M2=max(max(nir));
        
        %Normalize the sensor responses
        vis(:,:,1)=(vis(:,:,1)-m1(:,:,1))/(M1(:,:,1)-m1(:,:,1));
        vis(:,:,2)=(vis(:,:,2)-m1(:,:,2))/(M1(:,:,2)-m1(:,:,2));
        vis(:,:,3)=(vis(:,:,3)-m1(:,:,3))/(M1(:,:,3)-m1(:,:,3));
        nir(:,:,1)=(nir(:,:,1)-m2(:,:,1))/(M2(:,:,1)-m2(:,:,1));
        
        %Make sure that everything is between 0 and 1.
        vis(vis > 1.0) = 1.0;
        vis(vis < 0.0) = 0.0;
        nir(nir > 1.0) = 1.0;
        nir(nir < 0.0) = 0.0;
        
        %Compute grayscale image of visible image
        L = (vis(:,:,1) + vis(:,:,2) + vis(:,:,3))/3.0;
        
        %Compute dark maps of visible and NIR image
        L = L.^(1/gamma);
        D_vis = f(L, a, b);
        nir = nir.^(1/gamma);
        D_nir = f(nir, a, b);
        
        %Compute shadow candidate map
        D = D_vis.*D_nir;
        
        %Compute color to NIR ratios (use this only for unprocessed images).      
        if(useratios == 1)
            [m,n,c] = size(vis);
            Tk = zeros(m,n,c);
            Tk(:,:,1) = vis(:,:,1)./nir;
            Tk(:,:,2) = vis(:,:,2)./nir;
            Tk(:,:,3) = vis(:,:,3)./nir;
            
            T = min(max(Tk, [], 3), thresh);
            
            %Normalize color to NIR ratio
            T = T./thresh;
            
            %Compute shadow map
            U = (1-D).*(1-T);
        else
            %If we don't use the color to NIR ratio, the shadow map will be
            %given by the shadow candidates.
            U = (1-D);
        end
        %Compute initial number of histogram bins
        [a1,b1]=size(U);
        HIST_LENGTH= 1.6*ceil(log2(a1*b1)+1);

        try
            [~, TT] = valley_hist(U, floor(HIST_LENGTH));
        catch
            disp(['Image ' ': Valley could not be determined. Using value of 0.5']);
            TT = 0.5;
        end
        thres = TT;

        %Compute shadow mask by thresholding at theta
%         U_bin = U <= theta;
        shadow = U;
        
        
        %Save resulting shadow mask
%         namebase = filenames_vis(i).name(1:(regexp(filenames_vis(i).name, '_vis') - 1));
        %Write binary mask
%         imwrite(U_bin, [results_folder '/Ours/' namebase '_ours.png']);
        %Write shadow map
        %imwrite(U, [results_folder '/Ours/map_' namebase '_ours.png']);

%     disp(['Done with our method... Total time is: ' num2str(0) 's, average per image: ' num2str(sum(t_ours)/length(filenames_vis)) ' s, standard deviation: ' num2str(std(t_ours)) '.'])

