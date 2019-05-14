if verLessThan('matlab','9.4')
    error(['Uses R2018b introduced functions ', ...
        'Update MATLAB version.'])
end

clc;
close all;

fprintf('\nLoading saved models...')
% Loading required files
load('net_alexnet.mat')
load('net_dleaf.mat')
load('net_vgg16.mat')
load('SVM.mat')
load('pca_dim.mat')
fprintf('\tDone.')

dbPath = '/home/user/ElezCET/latest/test images';
dbPath = fullfile(dbPath,'*.jpg');
[fn,pn] = uigetfile(dbPath);

filename = fullfile(pn,fn);
fprintf('\nSelected image:\n\t:%s',filename);

fprintf('\nPreprocessing...')
im1 = imPreprocessor1(filename);
im2 = imPreprocessor2(filename);
im3 = imPreprocessor3(filename);
fprintf('\tDone.')

fprintf('\nExtracting features...')

layer = 'drop7';
feat1 = activations(net1,im1,layer,'OutputAs','rows');
feat1 = feat1*reducedDimension;

layer = 'drop7';
feat2 = activations(net2,im2,layer,'OutputAs','rows');
feat2 = feat2*reducedDimension1;

layer = 'fc2';
feat3 = activations(net3,im3,layer,'OutputAs','rows');
feat3 = feat3*reducedDimension3;

fprintf('\tDone.')


FFV_test = (feat1+ feat2 + feat3) / 3;


Y_pred = predict(SVM_Model,FFV_test);
pred = upper(string(Y_pred));
fprintf('\nPrediction:\n\t%s',pred)

imshow(imread(filename));
title(pred);