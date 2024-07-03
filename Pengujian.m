clc; clear; close all;

image_folder = 'datauji';
filenames = dir(fullfile(image_folder, '*.jpg'));
total_images = numel(filenames);

data_uji = zeros(12,total_images);

for n = 1:total_images
   full_name = fullfile(image_folder, filenames(n).name);
    Img = imread(full_name);
    Img = im2double(Img);
    
    %Konversi ke citra abu-abu
    grayImage = rgb2gray(Img);
    
    %Citra biner menggunakan ambang Otsu
    threshold = graythresh(grayImage);
    binaryImage = imbinarize(grayImage, threshold);
    
    
    % Morfologi Opening
    se = strel('disk', 5);
    openedImage = imopen(binaryImage, se);
    %  Segmentasi (hilangkan latar belakang)
    segmentedImage = Img;
    segmentedImage(repmat(openedImage, [1, 1, 3])) = 255;

    % Konversi ke Ruang Warna HSV dan YCbCr
    hsvImg = rgb2hsv(segmentedImage);
    YCbCrImg = rgb2ycbcr(segmentedImage);

    % Ekstraksi Ciri Warna RGB
    R = Img(:,:,1);
    G = Img(:,:,2);
    B = Img(:,:,3);
    
    CiriR = mean2(R);
    CiriG = mean2(G);
    CiriB = mean2(B);
    
    % Ekstraksi Ciri Warna HSV
    H = hsvImg(:,:,1);
    S = hsvImg(:,:,2);
    V = hsvImg(:,:,3);
    
    CiriH = mean2(H);
    CiriS = mean2(S);
    CiriV = mean2(V);
    
    % Ekstraksi Ciri Warna YCbCr
    Y = YCbCrImg(:,:,1);
    Cb = YCbCrImg(:,:,2);
    Cr = YCbCrImg(:,:,3);
    
    CiriY = mean2(Y);
    CiriCb = mean2(Cb);
    CiriCr = mean2(Cr);
    
    % Ekstraksi Ciri Tekstur Filter Gabor
    I = (rgb2gray(Img));
    wavelength = 4;
    orientation = 90;
    [mag,phase] = imgaborfilt(I,wavelength,orientation);
    
    H = imhist(mag)';
    H = H/sum(H);
    I = (0:255)/255;
    
    CiriMEAN = mean2(mag);
    CiriENT = -H*log2(H+eps)';
    CiriVAR = (I-CiriMEAN).^2*H';
    
    % Pembentukan data latih
    data_uji(1,n) = CiriR;
    data_uji(2,n) = CiriG;
    data_uji(3,n) = CiriB;
    data_uji(4,n) = CiriH;
    data_uji(5,n) = CiriS;
    data_uji(6,n) = CiriV;
    data_uji(7,n) = CiriY;
    data_uji(8,n) = CiriCb;
    data_uji(9,n) = CiriCr;
    data_uji(10,n) = CiriMEAN;
    data_uji(11,n) = CiriENT;
    data_uji(12,n) = CiriVAR;
end

% Inisialisasi data latih dan target
input = data_uji;
target = zeros(1,6);
target(:,1:2) = 1;
target(:,3:4) = 2;
target(:,5:6) = 3;

load net
output = round(sim(net,input));

% Display Confusion Matrix as a Chart
confMat = confusionmat(target(:), output(:));
figure;
confusionchart(confMat, {'Class 1', 'Class 2', 'Class 3'}, 'Title', 'Confusion Matrix for Training');

% Calculate accuracy
accuracy = sum(diag(confMat)) / sum(confMat(:)) * 100;
disp(['Accuracy: ' num2str(accuracy) '%']);
