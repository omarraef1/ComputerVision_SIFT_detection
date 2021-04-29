function hw9enw()

%map from gray images to colored ones

close all;
format compact;

%datacursormode on

slide1G = imread('slide1.pgm');
slide2G = imread('slide2.pgm');
slide3G = imread('slide3.pgm');
frame1G = imread('frame1.pgm');
frame2G = imread('frame2.pgm');
frame3G = imread('frame3.pgm');

slide1C = imread('slide1.tiff');
slide2C = imread('slide2.tiff');
slide3C = imread('slide3.tiff');
frame1C = imread('frame1.jpg');
frame2C = imread('frame2.jpg');
frame3C = imread('frame3.jpg');

slide1Cex = slide1C(:,:,1:3);
figure(1), imshowpair(slide1Cex, slide1G, 'montage');
slide2Cex = slide2C;
figure(2), imshowpair(slide2Cex, slide2G, 'montage');
slide3Cex = slide3C;
figure(3), imshowpair(slide3Cex, slide3G, 'montage');
frame1Cex = frame1C;
figure(4), imshowpair(frame1Cex, frame1G, 'montage');
frame2Cex = frame2C;
figure(5), imshowpair(frame2Cex, frame2G, 'montage');
frame3Cex = frame3C;
figure(6), imshowpair(frame3Cex, frame3G, 'montage');


whos('slide1G')
whos('slide2G')
whos('slide3G')
whos('frame1G')
whos('frame2G')
whos('frame3G')
whos('slide1C')
whos('slide2C')
whos('slide3C')
whos('frame1C')
whos('frame2C')
whos('frame3C')


%THIS COMMENT STUB STATES THAT 
%THIS CODE IS THE PROPERTY OF OMAR R.G. (UofA Student)


%___ keypoints ___%


slide1_kp = double(slide1G);
keyP = SIFT(slide1_kp,2, 4, 1.01);
keyPImg = slide1C(:,:,1:3);
figure(7), imshow(keyPImg)
hold on
for i = 1:128
    [x, y] = keyP{i}.coordinates();
    plot(y,x, 'y.');
    hold on
    th = 0:pi/50:2*pi;
    xunit = 7 * cos(th) + y;
    yunit = 7 * sin(th) + x;
    plot(xunit, yunit, 'b');
    hold on
end
hold off

frame1_kp = double(frame1G);
keyP2 = SIFT(frame1_kp,1, 5, 0.74);
whos('keyP2')
keyPImg2 = frame1C;
figure(8), imshow(keyPImg2)
hold on
for i = 1:128
    [x, y] = keyP2{i}.coordinates();
    plot(y,x, 'r.');
    hold on
    th = 0:pi/50:2*pi;
    xunit = 7 * cos(th) + y;
    yunit = 7 * sin(th) + x;
    plot(xunit, yunit, 'b');
    hold on
end
hold off



end

function Descriptors = SIFT(inputImage, Octaves, Scales, Sigma)
% This function is to extract sift features from a given image
    
    %% Setting Variables.
    Sigmas = sigmas(Octaves,Scales,Sigma);
    ContrastThreshhold = 7.68;
    rCurvature = 10;
    G = cell(1,Octaves); % Gaussians
    D = cell(1,Octaves); % DoG
    GO = cell(1,Octaves); % Gradient Orientation
    GM = cell(1,Octaves); % Gradient Scale
    P = [];
    Descriptors = {}; % Key Points

    %% Calculating Gaussians
    for o = 1:Octaves
        [row,col] = size(inputImage);
        temp = zeros(row,col,Scales);
        for s=1:Scales
            temp(:,:,s) = imgaussfilt(inputImage,Sigmas(o,s));
        end
        G(o) = {temp};
        inputImage = inputImage(2:2:end,2:2:end);
    end

    %% Calculating DoG
    for o=1:Octaves
        images = cell2mat(G(o));
        [row,col,Scales] = size(images);
        temp = zeros([row,col,Scales-1]);
        for s=1:Scales-1
            temp(:,:,s) = images(:,:,s+1) - images(:,:,s);
        end
        D(o) = {temp};
    end
    
    %% Calculating orientation of gradient in each scale
    for o = 1:Octaves
        images = cell2mat(G(o));
        [row,col,Scales] = size(images);
        tempO = zeros([row,col,Scales]);
        tempM = zeros([row,col,Scales]);
        for s = 1:Scales
            [tempM(:,:,s),tempO(:,:,s)] = imgradient(images(:,:,s));
        end
        GO(o) = {tempO};
        GM(o) = {tempM};
    end

    %% Extracting Key Points
    for o=1:Octaves
        images = cell2mat(D(o));
        GradientOrientations = cell2mat(GO(o));
        GradientMagnitudes = cell2mat(GM(o));
        [row,col,Scales] = size(images);
        for s=2:Scales-1
            % Weight for gradient vectors
            weights = gaussianKernel(Sigmas(o,s));
            radius = (length(weights)-1)/2;
            for y=14:col-12
                for x=14:row-12
                    sub = images(x-1:x+1,y-1:y+1,s-1:s+1);
                    if sub(2,2,2) > max([sub(1:13),sub(15:end)]) || sub(2,2,2) < min([sub(1:13),sub(15:end)])
                        % Getting rid of bad Key Points
                        if abs(sub(2,2,2)) < ContrastThreshhold
                            % Low contrast.
                            continue
                        else
                            % Calculating trace and determinant of hessian
                            % matrix.
                            Dxx = sub(1,2,2)+sub(3,2,2)-2*sub(2,2,2);
                            Dyy = sub(2,1,2)+sub(2,3,2)-2*sub(2,2,2);
                            Dxy = sub(1,1,2)+sub(3,3,2)-2*sub(1,3,2)-2*sub(3,1,2);
                            trace = Dxx+Dyy;
                            determinant = Dxx*Dyy-Dxy*Dxy;
                            curvature = trace*trace/determinant;
                            if curvature > (rCurvature+1)^2/rCurvature
                                % Not a corner.
                                continue
                            end
                        end
                        %% Calculating orientation and magnitude of pixels at key point vicinity
                        % Fixing overflow key points near corners and edges
                        % of image.
                        a=0;b=0;c=0;d=0;
                        if x-1-radius < 0;a = -(x-1-radius);end
                        if y-1-radius < 0;b = -(y-1-radius);end
                        if row-x-radius < 0;c = -(row-x-radius);end
                        if col-y-radius < 0;d = -(col-y-radius);end
                        tempMagnitude = GradientMagnitudes(x-radius+a:x+radius-c,y-radius+b:y+radius-d,s).*weights(1+a:end-c,1+b:end-d);
                        tempOrientation = GradientOrientations(x-radius+a:x+radius-c,y-radius+b:y+radius-d,s);
                        [wRows, wCols] = size(tempMagnitude);
                        % 36 bin histogram generation.
                        gHist = zeros(1,36);
                        for i = 1:wRows
                            for j = 1:wCols
                                % Converting orientation calculation window
                                temp = tempOrientation(i,j);
                                if temp < 0
                                    temp = 360 + temp;
                                end
                                bin = floor(temp/10)+1;
                                gHist(bin) = gHist(bin) + tempMagnitude(i,j);
                            end
                        end
                        %% Extracting keypoint coordinates
                        % TODO: Interpolation for X and Y value to get
                        % subpixel accuracy.
                        %% Extracting keypoint orientation
                        % Marking 80% Threshold
                        orientationThreshold = max(gHist(:))*4/5;
                        tempP = [];
                        for i=1:length(gHist)
                            if gHist(i) > orientationThreshold
                                % Connrection both ends of the histogram
                                % for interpolation
                                if i-1 <= 0
                                    X = 0:2;
                                    Y = gHist([36,1,2]);
                                elseif i+1 > 36
                                    X = 35:37;
                                    Y = gHist([35,36,1]);
                                else
                                    X = i-1:i+1;
                                    Y = gHist(i-1:i+1);
                                end
                                % interpolation of Orientation.
                                dir = interpolateExterma([X(1),Y(1)],[X(2),Y(2)],[X(3),Y(3)])*10; % Orientation
                                mag = gHist(i); % Size
                                % Filtering points with the same
                                % orientation.
                                if ismember(dir,tempP(5:6:end)) == false
                                    tempP = [tempP,x,y,o,s,dir,mag];
                                end
                            end
                        end
                        P = [P,tempP];
                    end
                end
            end
        end
    end
    %% Creating feature Descriptors
    % TODO: Extract Descriptors
    weights = gaussianKernel(Sigmas(o,s),13);
    weights = weights(1:end-1,1:end-1);
    for i = 1:6:length(P)
        x = P(i);
        y = P(i+1);
        oct = P(i+2);
        scl = P(i+3);
        dir = P(i+4);
        mag = P(i+5);
        directions = cell2mat(GO(oct));
        directions = directions(x-13:x+12,y-13:y+12,scl);
        magnitudes = cell2mat(GM(oct));
        magnitudes = magnitudes(x-13:x+12,y-13:y+12,scl).*weights;
        descriptor = [];
        for m = 5:4:20
            for n = 5:4:20
                hist = zeros(1,8);
                for o = 0:3
                    for p = 0:3
                        [newx,newy] = rotateCoordinates(m+o,n+p,13,13,-dir);
                        % Creating 8 bin histogram.
                        hist(categorizeDirection8(directions(newx,newy))) = magnitudes(newx,newy);
                    end
                end
                descriptor = [descriptor, hist];
            end
        end
        descriptor = descriptor ./ norm(descriptor,2);
        for j =1:128
            if descriptor(j) > 0.2
                descriptor(j) = 0.2;
            end
        end
        descriptor = descriptor ./ norm(descriptor,2);
        % Creating keypoint object
        kp = KeyPoint;
        kp.Coordinates = [x*2^(oct-1),y*2^(oct-1)];
        kp.Magnitude = mag;
        kp.Direction = dir;
        kp.Descriptor = descriptor;
        kp.Octave = oct;
        kp.Scale = scl;
        Descriptors(end+1) = {kp};
    end
end
%% Function to extract Sigma values
function matrix = sigmas(octave,scale,sigma)
% Function to calculate Sigma values for different Gaussians
    matrix = zeros(octave,scale);
    k = sqrt(2);
    for i=1:octave
        for j=1:scale
            matrix(i,j) = i*k^(j-1)*sigma;
        end
    end
end
%% Calculating Gaussian value given SD
function result = gaussianKernel(SD, Radius)
% Returns a gaussian kernet
% By default a radius will be chosen to so kernel covers 99.7 % of data.
    if nargin < 2
        Radius = ceil(3*SD);
    end
    side = 2*Radius+1;
    result = zeros(side);
    for i = 1:side
        for j = 1:side
            x = i-(Radius+1);
            y = j-(Radius+1);
            result(i,j)=(x^2+y^2)^0.5;
        end
    end
    result = exp(-(result .^ 2) / (2 * SD * SD));
    result = result / sum(result(:));
end

%% Interpolation - Fiting a parabola into 3 points and extracting more exact Exterma
function exterma = interpolateExterma(X, Y, Z)
% Exterpolation and Exterma extraction
% Each input is an array with 2 values, t and f(t).
    exterma = Y(1)+...
        ((X(2)-Y(2))*(Z(1)-Y(1))^2 - (Z(2)-Y(2))*(Y(1)-X(1))^2)...
        /(2*(X(2)-Y(2))*(Z(1)-Y(1)) + (Z(2)-Y(2))*(Y(1)-X(1)));
end

%% Function to assign bins to orientations
% 8 bin assignment
function bin = categorizeDirection8(Direction)
    if Direction <= 22.5 && Direction > -22.5
        bin = 1;
    elseif Direction <= 67.5 && Direction > 22.5
        bin = 2;
    elseif Direction <= 112.5 && Direction > 67.5
        bin = 3;
    elseif Direction <= 157.5 && Direction > 112.5
        bin = 4;
    elseif Direction <= -157.5 || Direction > 157.5
        bin = 5;
    elseif Direction <= -112.5 && Direction > -157.5
        bin = 6;
    elseif Direction <= -67.5 && Direction > -112.5
        bin = 7;
    elseif Direction <= -22.5 && Direction > -67.5
        bin = 8;
    end
end

%% Rotating coordinates
function [x,y] =  rotateCoordinates(x, y, originx, originy, dir)
% Rotating a pixel around an origins
    p = [x,y,1]';
    translate = [1,0,-originx;0,1,-originy;0,0,1];
    rotate = [cosd(dir),-sind(dir),0;sind(dir),cosd(dir),0;0,0,1];
    translateBack = [1,0,originx;0,1,originy;0,0,1];
    p = translateBack*rotate*translate*p;
    x = floor(p(1));y = floor(p(2));
end

function image = SIFTKeypointVisualizer(Image, KeyPoints)
% This function creates an Image to visualize the SIFT features
% Input image should be gray scale.

    % Creating color image.
    image = cat(3, Image, Image, Image);
    for i=1:length(KeyPoints)
        [x, y] = KeyPoints{i}.coordinates();
        circle = [y,x,5];
        % adding circles to key point locations
        image = insertShape(image,'circle',circle,'LineWidth',1,'color',[255,0,0],'SmoothEdges',false);
    end
    for i=1:length(KeyPoints)
        [x, y] = KeyPoints{i}.coordinates();
        dir = KeyPoints{i}.direction();
        line = [y,x,y+10*sind(dir),x+10*cosd(dir)];
        % adding lines to key point locations
        image = insertShape(image,'line',line,'LineWidth',1,'color',[0,0,255]);
    end
    for i=1:6:length(KeyPoints)
        [x, y] = KeyPoints{i}.coordinates();
        % Distinguishing key point location with green dots
        image(x,y,:) = [0,255,0];
    end
end
