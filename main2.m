close all 
clear
clc

imds = imageDatastore('images\', ...
        'IncludeSubfolders',true,'LabelSource','foldernames');

[imdstrain, imdsvalid]=splitEachLabel(imds,.8,'randomize');
  
     
CountLabel = imds.countEachLabel

layers = [
    imageInputLayer([48 48  1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(7)
    softmaxLayer
    classificationLayer]; 

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',10, ...
    'Verbose',true, ...
    'Plots','training-progress');

convnet = trainNetwork(imdstrain,layers,options);

YPred = classify(convnet,imdsvalid);
YValidation = imdsvalid.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation)

plotconfusion(YValidation,YPred)


a=read(imdsvalid  );
class=classify(convnet,a)
figure
imshow(a)
title(string(class))


%
clear camera
figure
camera = webcam(1) % select your webcam
while true   
    im = camera.snapshot;     
    picture=rgb2gray(im);% Take a picture    
    picture = imresize(picture,[224 224]);  % Resize the picture
    label = classify(convnet, picture);        % Classify the picture
    image(im);     % Show the picture
    title(char(label)); % Show the label
    drawnow;   
end

