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
% Layers_MLP = [
%     imageInputLayer(input_size)    
%     fullyConnectedLayer(prod(input_size))
%     reluLayer()
%     fullyConnectedLayer(prod(input_size))
%     reluLayer()
%     fullyConnectedLayer(output_size, 'Name', 'Output')
%     regressionLayer()
% ];
% analyzeNetwork(Layers_MLP)

%% create VGG neural network - Aufgabe 6
Layers_VGG= [
    imageInputLayer(input_size)
    % first block
    convolution2dLayer(3, 64, "Padding", "same")
    reluLayer
    convolution2dLayer(3, 64, "Padding", "same")
    reluLayer
    maxPooling2dLayer(2, "Stride", 2)
    fullyConnectedLayer(output_size, 'Name', 'Output')
    regressionLayer()
];
analyzeNetwork(Layers_VGG)

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
[trainedNet, info] = trainNetwork(x_train, y_train, Layers_VGG, options);
ids = 0:50:size(info.ValidationRMSE, 2);
ids(1) = 1;
figure;plot(info.TrainingRMSE);hold on;plot(ids, info.ValidationRMSE(ids));
title('RMSE'); legend('training', 'validation'); xlabel('Minibatch');
figure;plot(info.TrainingLoss);hold on;plot(ids, info.ValidationLoss(ids));
title('Loss'); legend('training', 'validation'); xlabel('Minibatch');

%% Test Network  - Aufgabe 4
% use command "predict"
preds = predict(trainedNet, test_imgs);

% reconstruct field distribution
rebuild_imgs = mmf_rebuilt_image(preds, test_imgs, ceil(size(y_train, 2) / 2));

%%  Visualization results  - Aufgabe 5
% calculate Correlation between the ground truth and reconstruction
vis_test_imgs = squeeze(test_imgs);
vis_rebuild_imgs = squeeze(rebuild_imgs);
corr_vals = calc_corr(abs(vis_test_imgs), abs(vis_rebuild_imgs));

% calculate mean and std
corr_mean = mean(corr_vals);
corr_std = std(corr_vals);

figure; plot(corr_vals, ':'); hold on; 
line([1 size(corr_vals, 1)], [corr_mean corr_mean]);
% std_lines = [corr_mean - 3*corr_std, corr_mean + 3*corr_std];
% line([1 size(corr_vals, 1)], [std_lines(1) std_lines(1)]);
% line([1 size(corr_vals, 1)], [std_lines(2) std_lines(2)]);

figure; boxplot(corr_vals);
figure; histogram(corr_vals);

% calulate relative error of ampplitude and phase 
rel_cplx_err = vis_rebuild_imgs ./ vis_test_imgs;
rel_amp_err = abs(rel_cplx_err);
rel_phs_err = angle(rel_cplx_err);

mean_amp_err = mean(rel_amp_err, 3);
mean_phs_err = mean(rel_phs_err, 3);
figure;imagesc(mean_amp_err);title('Mean relative amp error');
figure;imagesc(mean_phs_err);title('Mean relative phs error');

%% save model
clear train_imgs test_imgs train_labels test_labels
save(fullfile('models', strcat(datestr(datetime, 'mm-dd-yyyy_HH-MM'), '.mat')));
disp('Model and Workspace saved')
