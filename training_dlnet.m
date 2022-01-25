% Training of dlnet-type neural network for mode decomposition

clear all
close all
%% load dataset
%  1. load the dataset

%  2. define the input and output size for neural network

height = 32;
width = 32;
channels = 1;

%% create VGG neural network - Aufgabe 7
inputSize = [height width channels];

Layers_VGG = [imageInputLayer(inputSize)];
for i = 1:3
    Layers_VGG = [
        Layers_VGG
        % ein Block
        convolution2dLayer(3, 64, 'name', "conv1_1", "Padding", "same")
        reluLayer('Name', "relu1_1")
        convolution2dLayer(3, 64, 'Name', "conv1_2", "Padding", "same")
        reluLayer('Name', "relu1_2")
        maxPooling2dLayer(2, "Stride", 2, 'Name', "pooling1")
    ]
end
fullyConnectedLayer(256, 'Name', "fc1")
fullyConnectedLayer(128, 'Name', "fc2")
fullyConnectedLayer(output_size, 'Name', "fc_output")

% use command dlnetwork()
dlnet = dlnetwork(Layers_VGG);

%% learnable parameters transfer  - Aufgabe 8 
% use Transfer Learning


%% Training network  - Aufgabe 7
% define hyperparameters
% miniBatchSize = 128;
% numEpochs = 10;
% learnRate = 0.001;
numObservations = size(XTrain,4);   
numIterationsPerEpoch = floor(numObservations./miniBatchSize);
% executionEnvironment = "parallel";    

%Visualize the training progress in a plot.
plots = "training-progress";
% Train Network
if plots == "training-progress"
    figure
    lineLossTrain = animatedline('Color', [0.85 0.325 0.098]);
    ylim([0 inf])
    xlabel("Iteration")
    ylabel("Loss")
    grid on
end
iteration = 0;
% Train Network
% Initialize the average gradients and squared average gradients.
averageGrad = [];
averageSqGrad = [];
for epoch = 1:numEpochs
    dlX= dlarray(single(X));
    Y = single(labels);
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        
        % 1. Read mini-batch of data and convert the labels to dummy
        % variables.

        
        % 2. Convert mini-batch of data to a dlarray.
        
        % If training on a GPU, then convert data to a gpuArray.

        % 3. Evaluate the model gradients and loss using the
        % modelGradients() and dlfeval()

        % 4. Update the network parameters using the Adam optimizer.

        % Display the training progress.
        if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain,iteration,double(gather(extractdata(loss))))
            title("Epoch: " + epoch + ", Elapsed: " + string(D) + ", Loss: " + num2str(double(gather(extractdata(loss)))));
            drawnow
        end
    end
end
%% Test Network
% transfer data to dlarray
% use command "predict"
% use command "extractdata" to extract data from dlarray

% reconstruct field distribution
% [] = mmf_rebuilt_image();

%%  Visualization results
% calculate Correlation between the ground truth and reconstruction
% calculate std
% plot()
% calulate relative error of ampplitude and phase 


%% save model
