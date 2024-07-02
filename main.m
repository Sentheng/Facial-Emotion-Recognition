close all 
clear
clc

% Image Datastore
imds = imageDatastore('Img', ...
        'IncludeSubfolders',true,'LabelSource','foldernames');

% Split the datastore into training and validation sets
[imdstrain, imdsvalid] = splitEachLabel(imds,.8,'randomize');

% Display the count for each label
CountLabel = imds.countEachLabel;

% Define the network layers
layers = [
    imageInputLayer([224 224 1])

    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 256, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer];

% Update the training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 10, ...
    'Shuffle', 'every-epoch', ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress',...
    'MiniBatchSize', 32);

covnet = trainNetwork(imdstrain, layers, options);

% Predict the labels of the validation data
YPred = classify(covnet,imdsvalid);
YValidation = imdsvalid.Labels;

% Compute the accuracy of the predictions
accuracy = sum(YPred == YValidation)/numel(YValidation);

% Display the confusion matrix figure; 
% New figure for the confusion matrix
figure
plotconfusion(YValidation,YPred);

% Display an example image with its predicted label figure; 
% New figure for the image display
a = read(imdsvalid);
class = classify(covnet, a);

figure
imshow(a)
title(string(class))


%
% clear camera
figure
camera = webcam(1) % select your webcam
while true   
    im = camera.snapshot;     
    picture=rgb2gray(im);% Take a picture    
    picture = imresize(picture,[224 224]);  % Resize the picture
    label = classify(covnet, picture);        % Classify the picture
    image(im);     % Show the picture
    title(char(label)); % Show the label
    drawnow;   
end
