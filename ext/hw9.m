function hw9()

close all;

%%% PART 1 %%%%

%%   SLIDE FRAME 1
%
slide1 = imread('slide1.tiff');
frame1 = imread('frame1.jpg');
slide1g = imread('slide1.pgm');
frame1g = imread('frame1.pgm');
slide1 = slide1(:,:,1:3);
whos('slide1')
whos('frame1')
whos('slide1g')
whos('frame1g')


[ks1row, ks1col, ks1scale, ks1ori] = computesift(slide1g);
[kf1row, kf1col, kf1scale, kf1ori] = computesift(frame1g);

for i = 1:size(ks1ori)
    if ks1row(i) == 3 && ks1col(i) == 3
        zz = ks1ori;
        zz
    end
end


%THIS COMMENT STUB STATES THAT 
%THIS CODE IS THE PROPERTY OF OMAR R.G. (UofA Student)

whos('ks1row')
whos('kf1row')
%{
figure, imshow('frame2.jpg')
hold on
for i = 1:size(ks1ori)
for j = 1:size(kf1ori)
    if (ks1ori(i)==kf1ori(j) && ks1scale(i)==kf1scale(j))
        plot(kf1col(j), kf1row(j), 'r.');
        break
    end
end
end
hold off
%}
%app = [x y];
figure, imshowpair(slide1,frame1, 'montage')
datacursormode on
hold on
count = 0;
countyes = 0;
for i = 1:size(ks1ori)
for j = 1:size(kf1ori)
    if (ks1ori(i)==kf1ori(j) && ks1scale(i)==kf1scale(j))
        count = count + 1;
        x = [ks1row(i) kf1row(j)];
        y = [ks1col(i) (kf1col(j)+360)];
        if(count < 4)
        plot(y, x, '-r');
        count = 0;
        break
        end
    end
end
end
hold off
%}

%% SLIDE FRAME 2
%
slide2 = imread('slide2.tiff');
frame2 = imread('frame2.jpg');
slide2g = imread('slide2.pgm');
frame2g = imread('frame2.pgm');
whos('slide2')
whos('frame2')
whos('slide2g')
whos('frame2g')


[ks2row, ks2col, ks2scale, ks2ori] = computesift(slide2g);
[kf2row, kf2col, kf2scale, kf2ori] = computesift(frame2g);

whos('ks2row')
whos('kf2row')
%{
figure, imshow('frame2.jpg')
hold on
for i = 1:size(ks1ori)
for j = 1:size(kf1ori)
    if (ks1ori(i)==kf1ori(j) && ks1scale(i)==kf1scale(j))
        plot(kf1col(j), kf1row(j), 'r.');
        break
    end
end
end
hold off
%}
%app = [x y];
figure, imshowpair(slide2,frame2, 'montage')
hold on
count = 0;
countyes = 0;
for i = 1:size(ks2ori)
for j = 1:size(kf2ori)
    if (ks2ori(i)==kf2ori(j) && ks2scale(i)==kf2scale(j))
        count = count + 1;
        x = [ks2row(i) kf2row(j)];
        y = [ks2col(i) (kf2col(j)+352)];
        if(count < 4)
        plot(y, x, '-r');
        count = 0;
        break
        end
    end
end
end
hold off
%}
%% SLIDE FRAME 3
%
slide3 = imread('slide3.tiff');
frame3 = imread('frame3.jpg');
slide3g = imread('slide3.pgm');
frame3g = imread('frame3.pgm');
whos('slide3')
whos('frame3')
whos('slide3g')
whos('frame3g')


[ks3row, ks3col, ks3scale, ks3ori] = computesift(slide3);
[kf3row, kf3col, kf3scale, kf3ori] = computesift(frame3);

whos('ks3row')
whos('kf3row')
%{
figure, imshow('frame2.jpg')
hold on
for i = 1:size(ks1ori)
for j = 1:size(kf1ori)
    if (ks1ori(i)==kf1ori(j) && ks1scale(i)==kf1scale(j))
        plot(kf1col(j), kf1row(j), 'r.');
        break
    end
end
end
hold off
%}
%app = [x y];
figure, imshowpair(slide3,frame3, 'montage')
hold on
count = 0;
for i = 1:size(ks3ori)
for j = 1:size(kf3ori)
    if (ks3ori(i)==kf3ori(j) && ks3scale(i)==kf3scale(j))
        count = count + 1;
        x = [ks3row(i) kf3row(j)];
        y = [ks3col(i) (kf3col(j)+352)];
        if(count < 4)
        plot(y, x, '-r');
        count = 0;
        break
        end
    end
end
end
hold off
%}

%%
%%% PART 2 %%%%

%% 


%%% SLIDE FRAME 1
%
%x = imread('slide1.tiff');
%y = imread('frame1.jpg');
hold off
figure, imshowpair(slide1,frame1, 'montage')
hold on
simCutoff = 0.008;
for i = 1:size(ks1ori)
for j = 1:size(kf1ori)
    if (ks1ori(i)==kf1ori(j) && ks1scale(i)==kf1scale(j))
        %count = count + 1;
        x = [ks1row(i) kf1row(j)];
        y = [ks1col(i) (kf1col(j)+360)];
       % if(count < 4)
            if(E_Similarity([ks1row(i) ks1col(i)], [kf1row(j) kf1col(j)]) <= simCutoff)
                E_Similarity([ks1row(i) ks1col(i)], [kf1row(j) kf1col(j)])
                plot(y, x, '-r');
            end
        %count = 0;
        break
       % end
    end
end
end
hold off
%}
%%
%%% SLIDE FRAME 2
%
hold off
figure, imshowpair(slide2,frame2, 'montage')
hold on
simCutoff = 0.006;
for i = 1:size(ks2ori)
for j = 1:size(kf2ori)
    if (ks2ori(i)==kf2ori(j) && ks2scale(i)==kf2scale(j))
        %count = count + 1;
        x = [ks2row(i) kf2row(j)];
        y = [ks2col(i) (kf2col(j)+352)];
       % if(count < 4)
            if(E_Similarity([ks2row(i) ks2col(i)], [kf2row(j) kf2col(j)]) <= simCutoff)
                E_Similarity([ks2row(i) ks2col(i)], [kf2row(j) kf2col(j)])
                plot(y, x, '-r');
            end
        %count = 0;
        break
       % end
    end
end
end
hold off
%}
%%
%%% SLIDE FRAME 3
%
figure, imshowpair(slide3,frame3, 'montage')
hold on
simCutoff = 0.008;
for i = 1:size(ks3ori)
for j = 1:size(kf3ori)
    if (ks3ori(i)==kf3ori(j) && ks3scale(i)==kf3scale(j))
        %count = count + 1;
        x = [ks3row(i) kf3row(j)];
        y = [ks3col(i) (kf3col(j)+352)];
       % if(count < 4)
            if(E_Similarity([ks3row(i) ks3col(i)], [kf3row(j) kf3col(j)]) <= simCutoff)
                E_Similarity([ks3row(i) ks3col(i)], [kf3row(j) kf3col(j)])
                plot(y, x, '-r');
            end
        %count = 0;
        break
       % end
    end
end
end
hold off
%}
%%
%%% PART 3 %%%%
%% SLIDE FRAME 1
%
hold off
figure, imshowpair(slide1,frame1, 'montage')
hold on
for i = 1:size(ks1ori)
for j = 1:size(kf1ori)
    
    if (ks1ori(i)==kf1ori(j) && ks1scale(i)==kf1scale(j))
        
        x = [ks1row(i) kf1row(j)];
        y = [ks1col(i) (kf1col(j)+360)];
        kx = [ks1row(i) ks1col(i)];
        ky = [kf1row(j) kf1col(j)];
        %E_Similarity(ks1ori(i), kf1ori(j))
        %E_Similarity(kx, ky)
    if(E_Similarity(kx, ky) <= 0.008)
          %E_Similarity(ks1ori(i), kf1ori(j))
          plot(y, x, '-r');
    end
    end
end
end
hold off
%}
%% SLIDE FRAME 2
%
figure, imshowpair(slide2,frame2, 'montage')
hold on
for i = 1:size(ks2ori)
for j = 1:size(kf2ori)
    
    if (ks2ori(i)==kf2ori(j) && ks2scale(i)==kf2scale(j))
        
        x = [ks2row(i) kf2row(j)];
        y = [ks2col(i) (kf2col(j)+352)];
        %diff = angdiff(ks2ori(i), kf2ori(j));
        kx = [ks2row(i) ks2col(i)];
        ky = [kf2row(j) kf2col(j)];
    if(E_Similarity(kx, ky) <= 0.004)
          %E_Similarity(ks2ori(i), kf2ori(j))
          plot(y, x, '-r');
    end
    end
end
end
hold off
%}
%% SLIDE FRAME 3
%
figure, imshowpair(slide3,frame3, 'montage')
hold on
for i = 1:size(ks3ori)
for j = 1:size(kf3ori)
    
    if (ks3ori(i)==kf3ori(j) && ks3scale(i)==kf3scale(j))
        
        x = [ks3row(i) kf3row(j)];
        y = [ks3col(i) (kf3col(j)+352)];
        kx = [ks3row(i) ks3col(i)];
        ky = [kf3row(j) kf3col(j)];
    if(E_Similarity(kx, ky) <= 0.003)
          %E_Similarity(ks2ori(i), kf2ori(j))
          plot(y, x, '-r');
    end
    end
end
end
hold off

%%
%part 4
%}

%% slide frame 1
%
hold off
figure, imshowpair(slide1,frame1, 'montage')
hold on
for i = 1:size(ks1ori)
for j = 1:size(kf1ori)
    
    if (ks1ori(i)==kf1ori(j) && ks1scale(i)==kf1scale(j))
        
        x = [ks1row(i) kf1row(j)];
        p1 = [ks1row(i) ks1col(i)];
        p2 = [kf1row(j) kf1col(j)];
        y = [ks1col(i) (kf1col(j)+360)];
        calc = (p1-p2).^2./p2;
        n1 = ks1row(i); N1 = kf1row(j);
        n2 = ks1col(i); N2 = kf1col(j);
        % Pooled estimate of proportion
       p0 = (n1+n2) / (N1+N2);
       % Expected counts under H0 (null hypothesis)
       n10 = N1 * p0;
       n20 = N2 * p0;
       % Chi-square test, by hand
       observed = [n1 N1-n1 n2 N2-n2];
       expected = [n10 N1-n10 n20 N2-n20];
       chi2stat = sum((observed-expected).^2 ./ expected);
       p = 1 - chi2cdf(chi2stat,1);
    if(p < 1 && p > 0.008)
        
        plot(y, x, '-r');       
    end
    end
end
end
hold off
%}

%% slide frame 2
%
figure, imshowpair(slide2,frame2, 'montage');
hold on;
for i = 1:size(ks2ori)
for j = 1:size(kf2ori)
    
    if (ks2ori(i)==kf2ori(j) && ks2scale(i)==kf2scale(j))
        
        x = [ks2row(i) kf2row(j)];
        p1 = [ks2row(i) ks2col(i)];
        p2 = [kf2row(j) kf2col(j)];
        y = [ks2col(i) (kf2col(j)+352)];
        calc = (p1-p2).^2./p2;
        n1 = ks2row(i); N1 = kf2row(j);
        n2 = ks2col(i); N2 = kf2col(j);
        % Pooled estimate of proportion
       p0 = (n1+n2) / (N1+N2);
       % Expected counts under H0 (null hypothesis)
       n10 = N1 * p0;
       n20 = N2 * p0;
       % Chi-square test, by hand
       observed = [n1 N1-n1 n2 N2-n2];
       expected = [n10 N1-n10 n20 N2-n20];
       chi2stat = sum((observed-expected).^2 ./ expected);
       p = 1 - chi2cdf(chi2stat,1);
    if(p < 1 && p > 0.006)
        
        plot(y, x, '-r');       
    end
    end
end
end
hold off;
%}

%% slide frame 3
%
figure, imshowpair(slide3,frame3, 'montage');
hold on;
for i = 1:size(ks3ori)
for j = 1:size(kf3ori)
    
    if (ks3ori(i)==kf3ori(j) && ks3scale(i)==kf3scale(j))
        
        x = [ks3row(i) kf3row(j)];
        p1 = [ks3row(i) ks3col(i)];
        p2 = [kf3row(j) kf3col(j)];
        y = [ks3col(i) (kf3col(j)+352)];
        calc = (p1-p2).^2./p2;
        n1 = ks3row(i); N1 = kf3row(j);
        n2 = ks3col(i); N2 = kf3col(j);
        % Pooled estimate of proportion
       p0 = (n1+n2) / (N1+N2);
       % Expected counts under H0 (null hypothesis)
       n10 = N1 * p0;
       n20 = N2 * p0;
       % Chi-square test, by hand
       observed = [n1 N1-n1 n2 N2-n2];
       expected = [n10 N1-n10 n20 N2-n20];
       chi2stat = sum((observed-expected).^2 ./ expected);
       p = 1 - chi2cdf(chi2stat,1);
    if(p < 1 && p > 0.6)
        plot(y, x, '-r');       
    end
    end
end
end
hold off;
%}

%%
%part 5

%%
%% SLIDE FRAME 1
%
figure, imshowpair(slide1,frame1, 'montage');
hold on;
for i = 1:size(ks1ori)
for j = 1:size(kf1ori)
    
    if (ks1ori(i)==kf1ori(j) && ks1scale(i)==kf1scale(j))
        
        x = [ks1row(i) kf1row(j)];
        y = [ks1col(i) (kf1col(j)+360)];
        
       if((knnsearch(ks1ori,ks1ori(i))/knnsearch(kf1ori,kf1ori(j)))<0.8)
          (knnsearch(ks1ori,ks1ori(i))/knnsearch(kf1ori,kf1ori(j)))
           plot(y, x, '-r');
       end
    end
end
end
hold off;
%}

%% SLIDE FRAME 2
%
figure, imshowpair(slide2,frame2, 'montage');
hold on;
for i = 1:size(ks2ori)
for j = 1:size(kf2ori)
    
    if (ks2ori(i)==kf2ori(j) && ks2scale(i)==kf2scale(j))
        
        x = [ks2row(i) kf2row(j)];
        y = [ks2col(i) (kf2col(j)+352)];
        
       if((knnsearch(ks2ori,ks2ori(i))/knnsearch(kf2ori,kf2ori(j)))<0.8)
          (knnsearch(ks2ori,ks2ori(i))/knnsearch(kf2ori,kf2ori(j)))
           plot(y, x, '-r');
       end
    end
end
end
hold off;
%}

%% SLIDE FRAME 3
%
figure, imshowpair(slide3,frame3, 'montage');
hold on;
for i = 1:size(ks3ori)
for j = 1:size(kf3ori)
    
    if (ks3ori(i)==kf3ori(j) && ks3scale(i)==kf3scale(j))
        
        x = [ks3row(i) kf3row(j)];
        y = [ks3col(i) (kf3col(j)+352)];
        
       if((knnsearch(ks3ori,ks3ori(i))/knnsearch(kf3ori,kf3ori(j)))<0.8)
          %(knnsearch(ks3ori,ks3ori(i))/knnsearch(kf3ori,kf3ori(j)))
           plot(y, x, '-r');
       end
    end
end
end
hold off;
%}
%% part 6
% using euclidean distance from part 2

%%
%
s1f1 = 0;
s2f1 = 0;
s3f1 = 0;
s1f2 = 0;
s2f2 = 0;
s3f2 = 0;
s1f3 = 0;
s2f3 = 0;
s3f3 = 0;

% SLIDE1 FRAME1 
%
simCutoff = 0.007;
for i = 1:size(ks1ori)
for j = 1:size(kf1ori)
    if (ks1ori(i)==kf1ori(j) && ks1scale(i)==kf1scale(j))
            if(E_Similarity([ks1row(i) ks1col(i)], [kf1row(j) kf1col(j)]) <= simCutoff)
                s1f1 = s1f1 + 1;
            end
        break
    end
end
end
% SLIDE1 FRAME2
for i = 1:size(ks1ori)
for j = 1:size(kf2ori)
    if (ks1ori(i)==kf2ori(j) && ks1scale(i)==kf2scale(j))
            if(E_Similarity([ks1row(i) ks1col(i)], [kf2row(j) kf2col(j)]) <= simCutoff)
                s1f2 = s1f2 + 1;
            end
        break
    end
end
end
% SLIDE1 FRAME3
for i = 1:size(ks1ori)
for j = 1:size(kf3ori)
    if (ks1ori(i)==kf3ori(j) && ks1scale(i)==kf3scale(j))
            if(E_Similarity([ks1row(i) ks1col(i)], [kf3row(j) kf3col(j)]) <= simCutoff)
                s1f3 = s1f3 + 1;
            end
        break
    end
end
end
% SLIDE2 FRAME1
for i = 1:size(ks2ori)
for j = 1:size(kf1ori)
    if (ks2ori(i)==kf1ori(j) && ks2scale(i)==kf1scale(j))
            if(E_Similarity([ks2row(i) ks2col(i)], [kf1row(j) kf1col(j)]) <= simCutoff)
                s2f1 = s2f1 + 1;
            end
        break
    end
end
end
% SLIDE2 FRAME2
for i = 1:size(ks2ori)
for j = 1:size(kf2ori)
    if (ks2ori(i)==kf2ori(j) && ks2scale(i)==kf2scale(j))
            if(E_Similarity([ks2row(i) ks2col(i)], [kf2row(j) kf2col(j)]) <= simCutoff)
                s2f2 = s2f2 + 1;
            end
        break
    end
end
end
% SLIDE2 FRAME3
for i = 1:size(ks2ori)
for j = 1:size(kf3ori)
    if (ks2ori(i)==kf3ori(j) && ks2scale(i)==kf3scale(j))
            if(E_Similarity([ks2row(i) ks2col(i)], [kf3row(j) kf3col(j)]) <= simCutoff)
                s2f3 = s2f3 + 1;
            end
        break
    end
end
end
% SLIDE3 FRAME1
for i = 1:size(ks3ori)
for j = 1:size(kf1ori)
    if (ks3ori(i)==kf1ori(j) && ks3scale(i)==kf1scale(j))
            if(E_Similarity([ks3row(i) ks3col(i)], [kf1row(j) kf1col(j)]) <= simCutoff)
                s3f1 = s3f1 + 1;
            end
        break
    end
end
end
% SLIDE3 FRAME2
for i = 1:size(ks3ori)
for j = 1:size(kf2ori)
    if (ks3ori(i)==kf2ori(j) && ks3scale(i)==kf2scale(j))
            if(E_Similarity([ks3row(i) ks3col(i)], [kf2row(j) kf2col(j)]) <= simCutoff)
                s3f2 = s3f2 + 1;
            end
        break
    end
end
end
% SLIDE3 FRAME3
for i = 1:size(ks3ori)
for j = 1:size(kf3ori)
    if (ks3ori(i)==kf3ori(j) && ks3scale(i)==kf3scale(j))
            if(E_Similarity([ks3row(i) ks3col(i)], [kf3row(j) kf3col(j)]) <= simCutoff)
                s3f3 = s3f3 + 1;
            end
        break
    end
end
end
%}

threebythree = [s1f1 s2f1 s3f1; s1f2 s2f2 s3f2; s1f3 s2f3 s3f3];
threebythree

end


function S = E_Similarity(n1, n2) % 1 if same, 0 if different
    S = 1/(1+sqrt(sum((n1-n2).^2)));
end

function [krow, kcol, kscale, kori] = computesift(imgg)
image = imgg;
image2= image;
[Irows, Icols, chan]=size(image);
if(chan == 3)
    image = rgb2gray(image); 
end
I = double(image);
Sigma = 1.6;
k=sqrt(2);
DOG_Times = 4;
Octaves = 3;
Octave_outputs = spacescale( I,DOG_Times,Octaves,Sigma,k );
counter = 1;
r_tr = 10;
OriBins = 36;
winfactor  = 1.5 ;
for i=1:Octaves
    for j=1:DOG_Times
        if(j~=DOG_Times && j~=1) 
            current = Octave_outputs(i).DOG{j}.DOG;
            below = Octave_outputs(i).DOG{j-1}.DOG;
            above = Octave_outputs(i).DOG{j+1}.DOG;
            L = Octave_outputs(i).DOG{j}.L1;
            S = Octave_outputs(i).DOG{j}.S;
            [rows, cols]=size(Octave_outputs(i).DOG{j}.DOG);
            for r=1:rows
                for c=1:cols
                   if (c~=1 && c~=cols && r~=1 && r~=rows) 
                    X = [current(r,c) current(r+1,c) current(r-1,c) current(r,c+1) current(r,c-1) current(r+1,c+1) current(r-1,c-1) current(r+1,c-1) current(r-1,c+1) below(r,c) below(r+1,c) below(r-1,c) below(r,c+1) below(r,c-1) below(r+1,c+1) below(r-1,c-1) below(r+1,c-1) below(r-1,c+1) above(r,c) above(r+1,c) above(r-1,c) above(r,c+1) above(r,c-1) above(r+1,c+1) above(r-1,c-1) above(r+1,c-1) above(r-1,c+1)];
                    [max_X,I_max_X]= max(X);
                    [min_X,I_min_X]= min(X);
                    if(I_max_X == 1 || I_min_X == 1)
                        Dx= (current(r,c+1) - current(r,c-1))/2;
                        Dy= (current(r+1,c) - current(r-1,c))/2;
                        Dz= (above(r,c) - below(r,c))/2;
                        D_firstDerivative = [Dx; Dy; Dz];
                        Dxx = current(r,c-1) - 2*current(r,c) + current(r,c+1);
                        Dyy = current(r-1,c) - 2*current(r,c) + current(r+1,c);
                        Dzz = below(r,c) - 2*current(r,c) + above(r,c);
                        Dxy = ((current(r+1,c+1) - current(r+1,c-1)) - (current(r-1,c+1) - current(r-1,c-1))) / 4;
                        Dxz = ((above(r,c+1) - above(r,c-1)) - (below(r,c+1) - below(r,c-1))) / 4;
                        Dyz = ((above(r+1,c) - above(r-1,c)) - (below(r+1,c) - below(r-1,c))) / 4;
                        D_secondDerivative = [Dxx Dxy Dxz;Dxy Dyy Dyz;Dxz Dyz Dzz];
                        extremum = -inv(D_secondDerivative)*D_firstDerivative;
                        if(I_max_X == 1)
                            D=max_X;
                        else
                            D=min_X;
                        end
                        x=c;
                        y=r;
                        z=j;
                        if(extremum(1) > 0.5 && ceil(c+extremum(1))<cols)
                            extremum(1) = ceil(x+extremum(1));
                            x= extremum(1);
                        end
                        if(extremum(2) > 0.5 && ceil(r+extremum(2))<rows)
                            extremum(2) = ceil(y+extremum(2));
                            y= extremum(2);
                        end
                        if(extremum(3) > 0.5 && ceil(j+extremum(3))<=DOG_Times)
                            extremum(3) = ceil(z+extremum(3));
                            z= extremum(3);
                            L = Octave_outputs(i).DOG{z}.L1;
                            S = Octave_outputs(i).DOG{z}.S;
                        end 
                        D_extremum = D + 0.5 * transp(D_firstDerivative) * extremum;
                        Dxx = current(y,x-1) - 2*current(y,x) + current(y,x+1);
                        Dyy = current(y-1,x) - 2*current(y,x) + current(y+1,x);
                        Dxy = ((current(y+1,x+1) - current(y+1,x-1)) - (current(y-1,x+1) - current(y-1,x-1))) / 4;
                        if (abs(D_extremum) >= (0.03*255))
                            H = [Dxx Dxy;Dxy Dyy];
                            if ( (trace(H).^2)/det(H) < ((r_tr+1).^2)/r_tr ) 
                                sigma = winfactor * Sigma * (2.^(z/k));
                               radius = round(3*sigma);
                               hist = zeros(1,OriBins);
                               angle = atan((L(y,x+1)-L(y,x-1))/(L(y+1,x)-L(y-1,x)));
                               for new_r = y-radius:y+radius
                                for new_c = x-radius:x+radius
                                    if (new_r > 1 && new_c > 1 && new_r < rows - 2 && new_c < cols - 2)
                                        gradVal = sqrt(power(L(new_r+1,new_c)-L(new_r-1,new_c),2)+power(L(new_r,new_c+1)-L(new_r,new_c-1),2));
                                        distsq = (y - new_r).^2 + (x - new_c).^2;
                                        if (gradVal > 0.0  &&  distsq < radius * radius + 0.5)
                                            weight = exp(- distsq / (2.0 * sigma * sigma));
                                            angle = atan((L(new_r,new_c+1)-L(new_r,new_c-1))/(L(new_r+1,new_c)-L(new_r-1,new_c)));
                                            bin =  ceil(OriBins * (angle + pi + 0.001) / (2.0 * pi));
                                            bin = min(bin, OriBins);
                                            hist(bin) = hist(bin) + weight * gradVal;
                                        end
                                    end
                                end
                               end
                               [~, in] = max(hist);
                               angle = 2*pi*(in)/OriBins;
                               temp_Peaks(counter) = [struct('r_peak',y,'c_peak',x,'i_octave',i,'ori',angle,'S',S)];
                               counter = counter +1;
                            end   
                        end            
                    end
                   end
                end
            end
        end
    end
end

for i=1:length(temp_Peaks)
    c = temp_Peaks(i).c_peak * power(2,(temp_Peaks(i).i_octave)-1);
    r = temp_Peaks(i).r_peak * power(2,(temp_Peaks(i).i_octave)-1);
    locs(i,:) = [r c temp_Peaks(i).S temp_Peaks(i).ori];
end

image = image2;
computed_features = counter - 1
[krowt, kcolt, kscalet, korit] = computekeys(image, locs);
krow = krowt;
kcol = kcolt;
kscale = kscalet;
kori = korit;

end

function [row, col, scale, ori] = computekeys(image, locs)

figure,
%colormap('gray');
imshow(image)
datacursormode on
hold on
imsize = size(image);
for i = 1: size(locs,1)
    makeLine(imsize, locs(i,:), 0.0, 0.0, 1.0, 0.0);
    makeLine(imsize, locs(i,:), 0.85, 0.1, 1.0, 0.0);
    makeLine(imsize, locs(i,:), 0.85, -0.1, 1.0, 0.0);
end
for i = 1:size(locs,1)
    plot(locs(i,2), locs(i,1), 'y.');
    th = 0:pi/50:2*pi;
    xunit = 7 * cos(th) + locs(i,2);
    yunit = 7 * sin(th) + locs(i,1);
    plot(xunit, yunit, 'w');
end
hold off
row = locs(:, 1);
col = locs(:, 2);
scale = locs(:, 3);
ori = locs(:, 4);
end

function makeLine(imsize, keypoint, x1, y1, x2, y2)

len = 6 * keypoint(3);

s = sin(keypoint(4));
c = cos(keypoint(4));

r1 = keypoint(1) - len * (c * y1 + s * x1);
c1 = keypoint(2) + len * (- s * y1 + c * x1);
r2 = keypoint(1) - len * (c * y2 + s * x2);
c2 = keypoint(2) + len * (- s * y2 + c * x2);

line([c1 c2], [r1 r2], 'Color', 'c');
end

function Octave_outputs = spacescale( I_input,DOG_Times,Octaves,Sigma,k )

DOG = cell(1,DOG_Times);
counter = 0;

for j=1:Octaves
        GaussainSize = ceil(7*(Sigma * (k^counter)));
        H = fspecial('gaussian' , [GaussainSize GaussainSize], Sigma * (k^counter));
        if(j>1)
            I_input = I_input(1:2:end,1:2:end);
        end
    for i=1:DOG_Times
      L1 = conv2(I_input,H,'same');
      GaussainSize = ceil(7*(Sigma*(k^(i+counter))));
      H = fspecial('gaussian' , [GaussainSize GaussainSize], Sigma*(k^(i+counter)));
      L2 = conv2(I_input,H,'same');
      Octave_outputs(j).DOG{i} = struct('DOG',L2 - L1,'L2',L2,'L1',L1,'S',Sigma*(k^(i+counter)));
    end
    counter = counter + 1;
end

end
