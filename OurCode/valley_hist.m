% Function that takes the map image and outputs a thresholded version
% according to its histogram. Threshold occurs in the first valley.

function [I_out2, TT] = valley_hist(I_out, HIST_LENGTH)
[N, X] = hist(I_out(:), HIST_LENGTH); 
%Append the first signal twice to the left of the histogram, and the
%last twice to the right of the histogram.
N_ext = [N(1) N(1) N N(HIST_LENGTH) N(HIST_LENGTH)];
N1 = N_ext(1:length(N)) - N;
N2 = N_ext(2:length(N)+1) - N;
N3 = N_ext(4:length(N)+3) - N;
N4 = N_ext(5:length(N)+4) - N;

%Put N1-N4 into a matrix
N_tot = [N1; N2; N3; N4];
N_tot(N_tot==0) = 1; 
N_tot(N_tot< 0) = 0; 
N_tot(N_tot~=0) = 1;

S = sum(N);
C = cumsum(N)/S;

f2 = find(C > 0.95);

%We define a valley as being a "V" shape in the histogram with at least two
%consecutive decreasing values followed by 2 consecutive increasing values,
%in order to prevent local minima that are not valley shaped.
SS = sum(N_tot);
SS(f2) = 0;
SS(1) = 0;
SS(length(N)) = 0;

fv = find(SS==4);
f_safe = fv;
%keyboard
for i=1:length(fv)
    if(N_tot(2,fv(i)-1) == 1)&&(N_tot(3,fv(i)+1) == 1) %enforcing the "V" shape
        
    else
        fv(i)=0;
    end
end

fv = fv(find(fv));

% Threshold at valley
I_out2 = I_out;
if(isempty(fv))
    display(sprintf('image: fv empty. Length= %i', HIST_LENGTH));
    %If no such V-shape exist, increase the number of bins of the histogram
    %and reprocess the image.
    if(HIST_LENGTH < 100)
        [I_out2 TT] = valley_hist(I_out, round(HIST_LENGTH*1.1));       
    end
else
    TT = X(fv(1));

    TT_old = TT;
    [Nl, Xl] = hist(I_out(:), 256);
    
    candidates = find(Xl> X(fv(1)-1) & Xl < X(fv(1)+1));
    TT = Xl(candidates(find(Nl(candidates) == min(Nl(candidates)))));
    TT = TT(1);
    if(TT > 0.7)
        disp('Threshold above 0.7. Using finer histogram resolution...')
        [I_out2 TT] = valley_hist(I_out, round(HIST_LENGTH*1.1));
    else
    disp(['Threshold on low sample histogram: ' num2str(TT_old), ', finer threshold : ' num2str(TT)]);

    I_out2(I_out>TT) = 0;
    I_out2(I_out<=TT) = 1;
    end
end