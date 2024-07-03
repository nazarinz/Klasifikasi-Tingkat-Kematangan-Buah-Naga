clc; 
clear; 
close all;

image_folder = 'datalatih';
filenames = dir(fullfile(image_folder, '*.jpg'));
total_images = numel(filenames);

data_latih = zeros(12,total_images);

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
    data_latih(1,n) = CiriR;
    data_latih(2,n) = CiriG;
    data_latih(3,n) = CiriB;
    data_latih(4,n) = CiriH;
    data_latih(5,n) = CiriS;
    data_latih(6,n) = CiriV;
    data_latih(7,n) = CiriY;
    data_latih(8,n) = CiriCb;
    data_latih(9,n) = CiriCr;
    data_latih(10,n) = CiriMEAN;
    data_latih(11,n) = CiriENT;
    data_latih(12,n) = CiriVAR;
end

% Inisialisasi data latih dan target
input = data_latih;
target = zeros(1,30);
target(:,1:10) = 1; %Busuk
target(:,11:20) = 2; %Matang
target(:,21:30) = 3; %Mentah

% Definisikan arsitektur jaringan
hiddenLayerSizes = [10 20]; % Jumlah neuron dalam setiap lapisan tersembunyi

net = feedforwardnet(hiddenLayerSizes, 'trainlm');

% Atur fungsi aktivasi untuk setiap layer
net.layers{1}.transferFcn = 'tansig'; % Fungsi aktivasi untuk layer tersembunyi
net.layers{2}.transferFcn = 'logsig'; % Fungsi aktivasi untuk layer output

net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-6;
net = train(net, input, target);
output = round(sim(net, input));
save net.mat net

% Display Confusion Matrix as a Chart
confMat = confusionmat(target(:), output(:));
figure;
confusionchart(confMat, {'Class 1', 'Class 2', 'Class 3'}, 'Title', 'Confusion Matrix for Training');

% Calculate accuracy
accuracy = sum(diag(confMat)) / sum(confMat(:)) * 100;
disp(['Accuracy: ' num2str(accuracy) '%']);