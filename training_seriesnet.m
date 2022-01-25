% Training of seriesnet-type neural network for mode decomposition
clear all

%% load dataset
%  1. load the dataset
dataset = '5-modes_32x32_10k.mat';
load(fullfile('data', dataset));

val_split = floor(0.8 * length(train_imgs));
x_train = train_imgs(:, :, 1, 1:val_split);
y_train = train_labels(1:val_split, :);

x_val = train_imgs(:, :, 1, val_split+1:end);
y_val = train_labels(val_split+1:end, :);

%  2. define the input and output size for neural network
input_size = size(train_imgs, 1, 2, 3);
output_size = size(train_labels, 2);

%% create MLP neural network - Aufgabe 3
Layers_MLP = [
    imageInputLayer(input_size)    
    fullyConnectedLayer(prod(input_size))
    reluLayer()
    fullyConnectedLayer(prod(input_size))
    reluLayer()
    fullyConnectedLayer(output_size, 'Name', 'Output')
    regressionLayer()
];
% analyzeNetwork(Layers_MLP)

%% create VGG neural network - Aufgabe 6
% Layers_VGG= [
%     imageInputLayer(input_size)
%     % first block
%     convolution2dLayer(3, 64, "Padding", "same")
%     reluLayer
%     convolution2dLayer(3, 64, "Padding", "same")
%     reluLayer
%     maxPooling2dLayer(2, "Stride", 2)
%     fullyConnectedLayer(output_size, 'Name', 'Output')
%     regressionLayer()
% ];
% analyzeNetwork(Layers_VGG)

%% Training network
% define "trainingOptions"
options = trainingOptions('adam');
options.InitialLearnRate = 0.001;
options.MaxEpochs = 30;
options.MiniBatchSize = 128;
options.ValidationPatience = 5;
options.ValidationData = {x_val, y_val};
% options.ExecutionEnvironment = 'parallel';
% options.Plots = 'training-progress';

% training using "trainNetwork"
[trainedNet, info] = trainNetwork(x_train, y_train, Layers_MLP, options);

%% Test Network  - Aufgabe 4
% use command "predict"
preds = predict(trainedNet, test_imgs);

% reconstruct field distribution
rebuild_imgs = mmf_rebuilt_image(preds, test_imgs, ceil(size(y_train, 2) / 2));

%%  Visualization results  - Aufgabe 5
% calculate Correlation between the ground truth and reconstruction
% calculate std
% plot()
% calulate relative error of ampplitude and phase 


%% save model

