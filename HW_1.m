%% BBL 588 - HOMEWORK 1
close all, clear all
%% 1,2 Gray scale
img = imread('SunnyLake.bmp'); 
r = img(:,:,1);
g = img(:,:,2);
b = img(:,:,3);
gray = (r+g+b)./3; 
% gray2 = rgb2gray(img);
% gray3 = 0.2989 * r + 0.5870 * g + 0.1140 * b; % Normally
% corr2(gray,gray2)
figure,subplot 121,imshow(img); title("RGB"); 
subplot 122, imshow(gray); title("0.3*(R+G+B)");
%% 3 Histogram
h = imhist(gray);
figure,imhist(gray);
%% 4,5 Obtain binary image with threshold
T = mean(gray(:)); % I simply defined threshold value as a mean of the grayscale image.
binary = gray>T; % if pixel value above T, output is 1
binary_2 = gray<T; % if pixel value below T, output is 1
figure, subplot 121; imshow(binary); title("Pixels above T");
subplot 122; imshow(binary_2); title("Pixels below T");
%% 6 Gaussian noise
r_n1 = imnoise(r,'gaussian',0,1^2/255);
r_n5 = imnoise(r,'gaussian',0,5^2/255);
r_n10 = imnoise(r,'gaussian',0,10^2/255);
r_n20 = imnoise(r,'gaussian',0,20^2/255);

g_n1 = imnoise(g,'gaussian',0,1^2/255);
g_n5 = imnoise(g,'gaussian',0,5^2/255);
g_n10 = imnoise(g,'gaussian',0,10^2/255);
g_n20 = imnoise(g,'gaussian',0,20^2/255);

b_n1 = imnoise(b,'gaussian',0,1^2/255);
b_n5 = imnoise(b,'gaussian',0,5^2/255);
b_n10 = imnoise(b,'gaussian',0,10^2/255);
b_n20 = imnoise(b,'gaussian',0,20^2/255);

figure,subplot 221; imshow(r_n1); title("Std=1");
subplot 222; imshow(r_n5); title("Std=5");
subplot 223; imshow(r_n10); title("Std=10");
subplot 224; imshow(r_n20); title("Std=20");

figure,subplot 221; imshow(g_n1); title("Std=1");
subplot 222; imshow(g_n5); title("Std=5");
subplot 223; imshow(g_n10); title("Std=10");
subplot 224; imshow(g_n20); title("Std=20");

figure,subplot 221; imshow(b_n1); title("Std=1");
subplot 222; imshow(b_n5); title("Std=5");
subplot 223; imshow(b_n10); title("Std=10");
subplot 224; imshow(b_n20); title("Std=20");
%% 7 Grayscale images from noisy r+g+b images
I_1 = (r_n1+g_n1+b_n1)./3;
I_5 = (r_n5+g_n5+b_n5)./3;
I_10 = (r_n10+g_n10+b_n10)./3;
I_20 = (r_n20+g_n20+b_n20)./3;
figure,subplot 221; imshow(I_1); title("Std=1");
subplot 222; imshow(I_5); title("Std=5");
subplot 223; imshow(I_10); title("Std=10");
subplot 224; imshow(I_20); title("Std=20");
%% 8 Low Pass Filters
lpf_1 = 1/9*ones(3); % low pass filter 3x3
lpf_2 = 1/25*ones(5); % low pass filter 5x5
%% 8.1 3x3 filter 
f_I_1 = imfilter(I_1,lpf_1);
f_I_5 = imfilter(I_5,lpf_1);
f_I_10 = imfilter(I_10,lpf_1);
f_I_20 = imfilter(I_20,lpf_1);
figure,subplot 221; imshow(f_I_1); title("3x3 LPF Std=1");
subplot 222; imshow(f_I_5); title("3x3 LPF Std=5");
subplot 223; imshow(f_I_10); title("3x3 LPF Std=10");
subplot 224; imshow(f_I_20); title("3x3 LPF Std=20");
%% 8.2 5x5 filter
f2_I_1 = imfilter(I_1,lpf_2);
f2_I_5 = imfilter(I_5,lpf_2);
f2_I_10 = imfilter(I_10,lpf_2);
f2_I_20 = imfilter(I_20,lpf_2);
figure,subplot 221; imshow(f2_I_1); title("5x5 LPF Std=1");
subplot 222; imshow(f2_I_5); title("5x5 LPF Std=5");
subplot 223; imshow(f2_I_10); title("5x5 LPF Std=10");
subplot 224; imshow(f2_I_20); title("5x5 LPF Std=20");
%% 8.3 Gauss Filter 
g_I_1 = imgaussfilt(I_1); % Sigma value = 0.5 as default
g_I_5 = imgaussfilt(I_5);
g_I_10 = imgaussfilt(I_10);
g_I_20 = imgaussfilt(I_20);
figure,subplot 221; imshow(g_I_1); title("Gauss Filter Std=1");
subplot 222; imshow(g_I_5); title("Gauss Filter Std=5");
subplot 223; imshow(g_I_10); title("Gauss Filter Std=10");
subplot 224; imshow(g_I_20); title("Gauss Filter Std=20");
%% 9 High Pass Filter
hpf_1 = [-1,-1,-1;-1,8,-1;-1,-1,-1];
hpf_2 = [0.17,0.67,0.17;0.67,-3.33,0.67;0.17,0.67,0.17];
%% 9.1 Laplacian filter
fh_I_1 = imfilter(I_1,hpf_1);
fh_I_5 = imfilter(I_5,hpf_1);
fh_I_10 = imfilter(I_10,hpf_1);
fh_I_20 = imfilter(I_20,hpf_1);
figure,subplot 221; imshow(fh_I_1); title("3x3 HPF Std=1");
subplot 222; imshow(fh_I_5); title("3x3 HPF Std=5");
subplot 223; imshow(fh_I_10); title("3x3 HPF Std=10");
subplot 224; imshow(fh_I_20); title("3x3 HPF Std=20");
%% 9.2 Laplacian filter 2 
fh2_I_1 = imfilter(I_1,hpf_2);
fh2_I_5 = imfilter(I_5,hpf_2);
fh2_I_10 = imfilter(I_10,hpf_2);
fh2_I_20 = imfilter(I_20,hpf_2);
figure,subplot 221; imshow(fh2_I_1); title("3x3 HPF Std=1");
subplot 222; imshow(fh2_I_5); title("3x3 HPF Std=5");
subplot 223; imshow(fh2_I_10); title("3x3 HPF Std=10");
subplot 224; imshow(fh2_I_20); title("3x3 HPF Std=20");
%% 9.3 High boost Filter
% f=I(x,y)*(A-1)+hpf_filtered(x,y)
A = 1.2; % A>=1
fhb_1 = double(I_1).*(A-1)+double(fh_I_1);
fhb_5 = double(I_5).*(A-1)+double(fh_I_5);
fhb_10 = double(I_10).*(A-1)+double(fh_I_10);
fhb_20 = double(I_20).*(A-1)+double(fh_I_20);
figure,subplot 221; imshow(uint8(fhb_1)); title("HBF Std=1");
subplot 222; imshow(uint8(fhb_5)); title("HBF Std=5");
subplot 223; imshow(uint8(fhb_10)); title("HBF Std=10");
subplot 224; imshow(uint8(fhb_20)); title("HBF Std=20");
%% 10 Salt and pepper noisy image
salt_and_pepper_noisy_image = imread('Figure_1.png');
r2 = salt_and_pepper_noisy_image(:,:,1);
g2 = salt_and_pepper_noisy_image(:,:,2);
b2 = salt_and_pepper_noisy_image(:,:,3);
% We can remove this noise with median filter. So
Jr = medfilt2(r2); Jg = medfilt2(g2); Jb = medfilt2(b2);
medfilted_each_channel = cat(3, Jr, Jg, Jb);
figure,subplot 121; imshow(salt_and_pepper_noisy_image); title("Salt&Pepper");
subplot 122; imshow(medfilted_each_channel); title("3x3 Medfilt");

