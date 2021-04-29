function hw9n()

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


%THIS COMMENT STUB STATES THAT 
%THIS CODE IS THE PROPERTY OF OMAR R.G. (UofA Student)


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


%___ keypoints ___%


end