
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
showLayerGraphs = 1;
doTraining = 0;
%% Reading and Preparing Data
fprintf('\nReading and Preparing Data ...')
dbPath = 'C:\Users\pc\Documents\mca\s6\multiDCNN\animal_database';

pathCheck = dir(dbPath);
if isempty(pathCheck)
    errordlg('Error.Check Database Path');
    return
end

fprintf('\nData storage directory:\n\t %s\n',dbPath);

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

