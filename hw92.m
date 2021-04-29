function hw92()
close all;
source = imread('slide2.tiff');
input = imread('frame2.jpg');

% Initial parameter settings
num_octaves = 4;
num_scales = 5;
antialias_sigma = 0.5;
initSigma = sqrt(2);
contrast_threshold = 0.03;
r_curvature = 10;

% Convert images to grayscale if RGB
if (size(source,3))
   source = rgb2gray(source); 
end 
if (size(input,3))
   input = rgb2gray(input); 
end 

appended = append_images('LR', input,source);
% figure(1)
% imshow(appended);
clear appended


%%
tic
%=========================================================================%
%                    Scale Space and LoG approximation
%=========================================================================%

% Create a cell array to hold all the octaves and scales of the Scale Space
ScaleSpace = cell(num_octaves,num_scales);
DoG = cell(num_octaves,num_scales-1); % Scales 1, 2, 3, 4, 5 correspond to DoGs of scales 1, 2, 3, 4

% For octave 1, scale 1 image, source --> blur with sigma = 0.5 --> double size
source_antialiased = imfilter(source, fspecial('gaussian',[5 5], antialias_sigma));
ScaleSpace{1,1} =  imresize(source_antialiased, 2, 'bilinear');

for oct = 1:num_octaves
    sigma = initSigma;  % reset sigma for each octave    
    for sc = 1:num_scales-1
        sigma = sigma * 2^((sc-1)/2);
        % Apply blur to get next scale in same octave
        ScaleSpace{oct,sc+1} = imfilter(ScaleSpace{oct,sc}, fspecial('gaussian',[5 5], sigma));
        DoG{oct,sc} = ScaleSpace{oct,sc} - ScaleSpace{oct,sc+1};        
    end    
    
    % Create the next octave image if not reached the last octave
    % Next octave's first scale = prev octave's first scale downsized by half
    if (oct < num_octaves)
        ScaleSpace{oct+1,1} = imresize(ScaleSpace{oct,1}, 0.5, 'bilinear');   
    end    
end
%%
% Display the Scale Space images
app_SS = cell(num_octaves);
for oct = 1:num_octaves
    app_SS{oct} = append_images('LR',ScaleSpace{oct,1}, ScaleSpace{oct,2}, ScaleSpace{oct,3}, ScaleSpace{oct,4}, ScaleSpace{oct,5});
end
appended_SS = append_images('TB', app_SS{1}, app_SS{2}, app_SS{3}, app_SS{4});
% figure(2)
% imshow(appended_SS);


% Display the DoG images
app_DoGs = cell(num_octaves);
for oct = 1:num_octaves
    app_DoGs{oct} = append_images('LR',DoG{oct,1}, DoG{oct,2}, DoG{oct,3}, DoG{oct,4});
end
appended_DoGs = append_images('TB', app_DoGs{1}, app_DoGs{2}, app_DoGs{3}, app_DoGs{4});
% figure(3)
% imshow(appended_DoGs);

clear appended_DoGs appended_SS app_DoGs app_SS antialias_sigma initSigma source_antialiased sigma
toc
disp('    Created Scale Space');
   
%%
tic
%=========================================================================%
%                        Find Keypoints: DoG extrema
%=========================================================================%
%  n scales --> (n-1) DoGs --> (n-1)-2 = (n-3) DoGs with keypoints as 1st and last DoG scales in each octave do not have sufficient neighbors to be checked for extrema
% DoGs of Scales 2, 3, 4 correspond to DoG_Keypts of scales 1, 2
DoG_Keypts = cell(num_octaves,num_scales-3);

for oct = 1:num_octaves
    for sc = 2:num_scales-2
        DoG_Keypts{oct,sc-1} = DoG_extrema(DoG{oct,sc-1}, DoG{oct,sc}, DoG{oct,sc+1}); 
    end
end

toc
disp('    Found Keypoints = DoG extrema');


%THIS COMMENT STUB STATES THAT 
%THIS CODE IS THE PROPERTY OF OMAR R.G. (UofA Student)

%%
tic
%=========================================================================%
%       Filter Keypoints: Remove low contrast and edge keypoints
%=========================================================================%
% DoGs of Scales 2, 3, 4 correspond to DoG_Keypts of scales 1, 2

for oct = 1:num_octaves
    for sc = 1:num_scales-3
        
        [x,y] = find(DoG_Keypts{oct,sc});  % indices of the Keypoints
        num_keypts = size(find(DoG_Keypts{oct,sc}));  % number of Keypoints
        level = DoG{oct,sc+1};
        
        for k = 1:num_keypts
            x1 = x(k);
            y1 = y(k);
            % Discard low contrast points
            if (abs(level(x1+1,y1+1)) < contrast_threshold)
               DoG_Keypts{oct,sc}(x1,y1) = 0;
            % Discard extrema on edges
            else
               rx = x1+1;
               ry = y1+1;
               % Get the elements of the 2x2 Hessian Matrix
               fxx = level(rx-1,ry) + level(rx+1,ry) - 2*level(rx,ry);   % double derivate in x direction
               fyy = level(rx,ry-1) + level(rx,ry+1) - 2*level(rx,ry);   % double derivate in y direction
               fxy = level(rx-1,ry-1) + level(rx+1,ry+1) - level(rx-1,ry+1) - level(rx+1,ry-1); % Partial derivate in x and y direction
               % Find Trace and Determinant of this Hessian
               trace = fxx + fyy;
               deter = fxx*fyy - fxy*fxy;
               curvature = trace*trace/deter;
               curv_threshold = ((r_curvature+1)^2)/r_curvature;
               if (deter < 0 || curvature > curv_threshold)   % Reject edge points if curvature condition is not satisfied
                   DoG_Keypts{oct,sc}(x1,y1 )= 0;
               end
            end
        end
    end
end

clear level trace deter x1 y1 rx ry fxx fxy fyy curv_threshold curvature contrast_threshold r_curvature
toc
disp('    Eliminated Keypoints with low contrast or on edges');

%%
tic
%=========================================================================%
%                   Assign Orientations to Keypoints
%=========================================================================%
% Allocate memory for the gradient magnitude and orientations
grad_mag = cell(size(ScaleSpace));
grad_angle = cell(size(ScaleSpace));

% Assign orientations & magnitude to the Scale Space images (all points, not just keypoints)
for oct = 1:num_octaves
    for sc = 1:num_scales
        level = ScaleSpace{oct, sc};
        
        A = level(1:end-2,2:end-1);
        B = level(3:end, 2:end-1);
        C = level(2:end-1,1:end-2);
        D = level(3:end, 3:end);
        grad_mag{oct,sc} = sqrt(double(((A-B).^2) + ((C-D).^2)));
        grad_angle{oct,sc} = atan(double((A-B)./(C-D))); 
    end
end

toc
disp('    Assigned Orientations');

whos('DoG_Keypts')
ddd = DoG_Keypts(4);
ddd{1}
%figure(
for i=1:86
    for j = 1:64
        
    end
end
end

function appended = append_images(mode, varargin)
    % Function to append images Left-Right or Top-Bottom
    %       mode = 'LR' or 'TB' for left-right or top-bottom concatenation
    % Rows (or cols) at the bottom (or right) of the smaller images are zero padded

    % Create a copy of input images before messing with them 
    % (else would be pass by reference)
    images = cell(1,nargin); images = varargin; 

    % dim_array corresponds to rows if 'LR' and cols if 'TB'
    dim_array = zeros(1,nargin-1);
    for n = 1:(nargin-1)
        if strcmp(mode,'LR')
            dim_array(n) = size(images{n},1);
        else
            dim_array(n) = size(images{n},2);
        end   
    end
    max_dim = max(dim_array);
    appended = [];

    for n = 1:nargin-1
        switch mode 
            case 'LR'  % Pad zeros at the bottom of smaller images
                if size(images{n},1) < max_dim
                    images{n}(max_dim,1) = 0;
                end
                appended = [appended   images{n}];
            case 'TB'  % Pad zeros at the right of smaller images
                if size(images{n},2) < max_dim
                    images{n}(1,max_dim) = 0;
                end   
                appended = [appended ; images{n}];   
        end
    end
end

function extrema = DoG_extrema(top, current, down)
    % Function to find the extrema keypoints given 3 matrices
    % A pixel is a keypoint if it is the extremum of its 26 neighbors (8 in current, and 9 each in top and bottom)

    [sx, sy] = size(current);

    % Look for local maxima
    % Check the 8 neighbors around the pixel in the same level
    local_maxima = (current(2:sx-1,2:sy-1) > current(1:sx-2,1:sy-2)) & ...
                   (current(2:sx-1,2:sy-1) > current(1:sx-2,2:sy-1)) & ...
                   (current(2:sx-1,2:sy-1) > current(1:sx-2,3:sy)) & ...
                   (current(2:sx-1,2:sy-1) > current(2:sx-1,1:sy-2)) & ...
                   (current(2:sx-1,2:sy-1) > current(2:sx-1,3:sy)) & ...
                   (current(2:sx-1,2:sy-1) > current(3:sx,1:sy-2)) & ...
                   (current(2:sx-1,2:sy-1) > current(3:sx,2:sy-1)) & ...
                   (current(2:sx-1,2:sy-1) > current(3:sx,3:sy)) ;

    % Check the 9 neighbors in the level above it
    local_maxima = local_maxima & (current(2:sx-1,2:sy-1) > top(1:sx-2,1:sy-2)) & ...
                                  (current(2:sx-1,2:sy-1) > top(1:sx-2,2:sy-1)) & ...
                                  (current(2:sx-1,2:sy-1) > top(1:sx-2,3:sy)) & ...
                                  (current(2:sx-1,2:sy-1) > top(2:sx-1,1:sy-2)) & ...
                                  (current(2:sx-1,2:sy-1) > top(2:sx-1,2:sy-1)) & ...  % same pixel in top
                                  (current(2:sx-1,2:sy-1) > top(2:sx-1,3:sy)) & ...
                                  (current(2:sx-1,2:sy-1) > top(3:sx,1:sy-2)) & ...
                                  (current(2:sx-1,2:sy-1) > top(3:sx,2:sy-1)) & ...
                                  (current(2:sx-1,2:sy-1) > top(3:sx,3:sy));        

    % Check the 9 neighbors in the level below it                         
    local_maxima = local_maxima & (current(2:sx-1,2:sy-1) > down(1:sx-2,1:sy-2)) & ...
                                  (current(2:sx-1,2:sy-1) > down(1:sx-2,2:sy-1)) & ...
                                  (current(2:sx-1,2:sy-1) > down(1:sx-2,3:sy)) & ...
                                  (current(2:sx-1,2:sy-1) > down(2:sx-1,1:sy-2)) & ...
                                  (current(2:sx-1,2:sy-1) > down(2:sx-1,2:sy-1)) & ...  % same pixel in down
                                  (current(2:sx-1,2:sy-1) > down(2:sx-1,3:sy)) & ...
                                  (current(2:sx-1,2:sy-1) > down(3:sx,1:sy-2)) & ...
                                  (current(2:sx-1,2:sy-1) > down(3:sx,2:sy-1)) & ...
                                  (current(2:sx-1,2:sy-1) > down(3:sx,3:sy));

    % Look for local minima
    % Check the 8 neighbors around the pixel in the same level
    local_minima = (current(2:sx-1,2:sy-1) < current(1:sx-2,1:sy-2)) & ...
                   (current(2:sx-1,2:sy-1) < current(1:sx-2,2:sy-1)) & ...
                   (current(2:sx-1,2:sy-1) < current(1:sx-2,3:sy)) & ...
                   (current(2:sx-1,2:sy-1) < current(2:sx-1,1:sy-2)) & ...
                   (current(2:sx-1,2:sy-1) < current(2:sx-1,3:sy)) & ...
                   (current(2:sx-1,2:sy-1) < current(3:sx,1:sy-2)) & ...
                   (current(2:sx-1,2:sy-1) < current(3:sx,2:sy-1)) & ...
                   (current(2:sx-1,2:sy-1) < current(3:sx,3:sy)) ;

    % Check the 9 neighbors in the level above it
    local_minima = local_minima & (current(2:sx-1,2:sy-1) < top(1:sx-2,1:sy-2)) & ...
                                  (current(2:sx-1,2:sy-1) < top(1:sx-2,2:sy-1)) & ...
                                  (current(2:sx-1,2:sy-1) < top(1:sx-2,3:sy)) & ...
                                  (current(2:sx-1,2:sy-1) < top(2:sx-1,1:sy-2)) & ...
                                  (current(2:sx-1,2:sy-1) < top(2:sx-1,2:sy-1)) & ...  % same pixel in top
                                  (current(2:sx-1,2:sy-1) < top(2:sx-1,3:sy)) & ...
                                  (current(2:sx-1,2:sy-1) < top(3:sx,1:sy-2)) & ...
                                  (current(2:sx-1,2:sy-1) < top(3:sx,2:sy-1)) & ...
                                  (current(2:sx-1,2:sy-1) < top(3:sx,3:sy));        

    % Check the 9 neighbors in the level below it                         
    local_minima = local_minima & (current(2:sx-1,2:sy-1) < down(1:sx-2,1:sy-2)) & ...
                                  (current(2:sx-1,2:sy-1) < down(1:sx-2,2:sy-1)) & ...
                                  (current(2:sx-1,2:sy-1) < down(1:sx-2,3:sy)) & ...
                                  (current(2:sx-1,2:sy-1) < down(2:sx-1,1:sy-2)) & ...
                                  (current(2:sx-1,2:sy-1) < down(2:sx-1,2:sy-1)) & ...  % same pixel in down
                                  (current(2:sx-1,2:sy-1) < down(2:sx-1,3:sy)) & ...
                                  (current(2:sx-1,2:sy-1) < down(3:sx,1:sy-2)) & ...
                                  (current(2:sx-1,2:sy-1) < down(3:sx,2:sy-1)) & ...
                                  (current(2:sx-1,2:sy-1) < down(3:sx,3:sy));

    extrema = local_maxima | local_minima;
    whos('extrema')
end
