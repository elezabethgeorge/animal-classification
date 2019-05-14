%% Effect of fusing features from multiple DCNNarchitectures in image classification
% Initialisation

if verLessThan('matlab','9.4')
    error(['Uses R2018b introduced functions ', ...
        'Update MATLAB version.'])
end

clc;
fprintf('\nInitialising ...')
clearvars;
close all;
warning('off')

showImages = 0;
showLayerGraphs = 0;
doTraining = 1;
%% Reading and Preparing Data
fprintf('\nReading and Preparing Data ...')
dbPath = 'D:\Database\animal_database\modified';

pathCheck = dir(dbPath);
if isempty(pathCheck)
    errordlg('Error.Check Database Path');
    return
end

fprintf('\nData storage directory:\n\t %s',dbPath);

imds1 = imageDatastore(dbPath,...
    'IncludeSubfolders', true,...
    'LabelSource','foldernames',...
    'ReadFcn',@imPreprocessor1);

imds2 = imageDatastore(dbPath,...
    'IncludeSubfolders', true,...
    'LabelSource','foldernames',...
    'ReadFcn',@imPreprocessor2);

imds3 = imageDatastore(dbPath,...
    'IncludeSubfolders', true,...
    'LabelSource','foldernames',...
    'ReadFcn',@imPreprocessor3);

[traindb1,testdb1] = splitEachLabel(imds1,0.7,'randomized');
[traindb2,testdb2] = splitEachLabel(imds2,0.7,'randomized');
[traindb3,testdb3] = splitEachLabel(imds3,0.7,'randomized');

numclasses = numel(unique(imds1.Labels));

trainLabels1 = traindb1.Labels;
trainLabels2 = traindb2.Labels;
trainLabels3 = traindb3.Labels;

numTrainImages = length(traindb1.Files);
classNames = string(unique(trainLabels1));


db_details = countEachLabel(traindb1);
labels_unique = db_details.Label;
disp(db_details)

%% Display a sample of training images.
if showImages
    fprintf('\nDisplaying a sample of training images ...')
    figure;
    idx = randperm(numTrainImages,25);
    count = 1;c = {};
    for i = idx
        c{count} = readimage(traindb1,i);
        count = count + 1;
    end
    montage(c,'BorderSize',[1,1],...
        'BackgroundColor',[.3 .1 .2])
    title('Display a sample of training images')
end

%% vgg16
fprintf('\nvgg16:Creating network ...')
net1 = vgg16;

inputSize = net1.Layers(1).InputSize;
layersTransfer = net1.Layers(1:end-3);
layers1 = [
    layersTransfer
    fullyConnectedLayer(numclasses,'Name','fc8','WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer('Name','softmaxL')
    classificationLayer('Name','classi')];


lgraph = layerGraph(layers1);
if showLayerGraphs
    fprintf('\nvgg16:Plotting layerGraph ...')
    figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
    plot(lgraph);
end
fprintf('\nvgg16:Defining network properties...')

options = trainingOptions('sgdm', ...
    'MiniBatchSize',60, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

if doTraining
    tic
    fprintf('\nvgg16:Training the CNN network...')
    net1 = trainNetwork(traindb1,layers1,options);
    save net_vgg16.mat net1
    t1 = toc;
    fprintf('\nvgg16:Time taken for training:%f secs',t1)
    
    fprintf('\nvgg16:Extracting CNN features ...')
    layer = 'drop7';
    tic
    featuresTrain1 = activations(net1,traindb1,layer,'OutputAs','rows');
    t2 = toc;%11.4000hours
    fprintf('\nLevel_1 | vgg16:Time taken for train set feature extraction:%f secs',t2)
    
    fprintf('\nvgg16:Saving CNN features ...')
    save feat1.mat featuresTrain1
    
else
    fprintf('\nvgg16:Loading trained network...')
    load('net_vgg16.mat')
    
    fprintf('\nvgg16:Loading CNN features ...')
    load('feat1.mat')
end

%% alexnet
fprintf('\nalexnet:Creating network ...')
net2 = alexnet;

inputSize = net2.Layers(1).InputSize;
layersTransfer = net2.Layers(1:end-3);

layers2 = [
    layersTransfer
    fullyConnectedLayer(numclasses,...
    'Name','fc8',...
    'WeightLearnRateFactor',20,...
    'BiasLearnRateFactor',20)
    softmaxLayer('Name','softmaxL')
    classificationLayer('Name','classi')];

layersTransfer = net2.Layers(3:16);
layers2 = [
    net2.Layers(1)
    convolution2dLayer([7,7],96,'Stride',2,'Name','conv1')
    layersTransfer
    fullyConnectedLayer(1290,'Name','fc6')
    reluLayer('Name','relu6')
    net2.Layers(19)
    fullyConnectedLayer(1290,'Name','fc7')
    reluLayer('Name','relu7')
    net2.Layers(22)
    fullyConnectedLayer(numclasses,'Name','fc8')
    softmaxLayer('Name','prob')
    classificationLayer('Name','output')
    ];
lgraph = layerGraph(layers2);
if showLayerGraphs
    fprintf('\nalexnet:Plotting layerGraph ...')
    figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
    plot(lgraph);
    title('Alexnet Layers')
end
fprintf('\nalexnet:Defining network properties...')

options = trainingOptions('adam',...
    'Plots','training-progress',...
    'InitialLearnRate',0.00001,...
    'MaxEpochs',30,...
    'Shuffle','every-epoch',...
    'LearnRateSchedule','none',...
    'Verbose',true,...
    'VerboseFrequency',5 );
%     'LearnRateDropFactor',0.0001,...
%     'LearnRateDropPeriod',60,...


if doTraining
    tic
    fprintf('\nalexnet:Training the CNN network...')
    net2 = trainNetwork(traindb2,layers2,options);
    save net_alexnet.mat net2
    t2 = toc;
    fprintf('\nalexnet:Time taken for training:%f secs',t2)
    
    fprintf('\nalexnet:Extracting CNN features ...')
    layer = 'drop7';
    tic
    featuresTrain2 = activations(net2,traindb2,layer,'OutputAs','rows');
    t2 = toc;
    fprintf('\nalexnet:Time taken for train set feature extraction:%f secs',t2)
    
    fprintf('\nalexnetSaving CNN features ...')
    save feat2.mat featuresTrain2
    
else
    fprintf('\nalexnet:Loading trained network...')
    load('net_alexnet.mat')
    
    fprintf('\nalexnet:Loading CNN features ...')
    load('feat2.mat')
end

%% D_Leaf
fprintf('\nD_Leaf:Defining Deepnet Layers ...')
layers = [imageInputLayer([250 250 1],'Normalization','zerocenter','Name','input')
    convolution2dLayer([11,11],64,'Stride',4,'Name','CS1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','PS1')
    convolution2dLayer([5,5],96,'Stride',2,'Name','CS2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2,'Stride',2,'Name','PS2')
    convolution2dLayer([4,4],256,'Stride',1,'Name','CS3')
    reluLayer('Name','relu3')
    %     maxPooling2dLayer(2,'Stride',2,'Name','PS3')
    fullyConnectedLayer(1290,'Name','fc4')
    fullyConnectedLayer(1290,'Name','fc5')
    fullyConnectedLayer(numclasses,'Name','fc6')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','final')];

lgraph = layerGraph(layers);
figure
plot(lgraph)
title('D_Leaf layer')

fprintf('\nD_Leaf:Specifying Training Options...')
options = trainingOptions('adam',...
    'Plots','training-progress',...
    'InitialLearnRate',0.0001,...
    'MaxEpochs',60,...
    'Shuffle','every-epoch',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.0001,...
    'LearnRateDropPeriod',60,...
    'Verbose',true,...
    'VerboseFrequency',60 );
if doTraining
    fprintf('\nD_Leaf:Training the network ...\n')
    
    tic% to note down time
    
    net3 = trainNetwork(traindb3,layers,options);
    fprintf('\nD_Leaf:Training Completed.')
    
    fprintf('\nD_Leaf:Saving trained network for future use ...')
    save net_dleaf.mat net3
    
    t1 = toc;
    timeString = datestr(t1/(24*60*60), 'DD:HH:MM:SS');
    fprintf('\nD_Leaf:Time taken for training : %s\n',timeString)
    
    
    
    layer = 'fc6';
    featuresTrain3 = activations(net3,traindb3,layer,'OutputAs','rows');
    save feat3.mat featuresTrain3
else
    fprintf('\nD_Leaf:Loading trained models...\n')
    load net_dleaf.mat
    load feat3.mat
end
