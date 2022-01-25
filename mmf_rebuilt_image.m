function [Image_data_complex, complex_vector_N] = mmf_rebuilt_image(pred_vectors, ground_truth, number_of_modes)
% Input>
% pred_vectorts: the prediction from neural network
% ground_truth: the correct amplitude distributions
% Outpu>
% Image_data_complex: reconstructed complex distribution 
% complex_vector_N: complex mode weights for all test data

if number_of_modes == 3   
    load('phase_variant_3.mat')
    load('mmf_3modes_32.mat')
    mmf_modes = mmf_3modes_32;
elseif number_of_modes == 5
    load('phase_variant_5.mat')
    load('mmf_5modes_32.mat')
    mmf_modes = mmf_5modes_32;
else
    errID = 'myComponent:inputError';
    msgtext = sprintf('No file for %d modes', number_of_modes);
    ME = MException(errID,msgtext);
    throw(ME)
end

image_size = size(mmf_modes, 1);

number_of_test_images = size(pred_vectors,1);
ground_truth = squeeze(ground_truth);

%% read mode weights from predicted vectors
% read amplitude weigths
amplitude_vector = pred_vectors(:,1:number_of_modes);
% read cos(phase) 
phase_vector = pred_vectors(:,(number_of_modes+1):(2*number_of_modes-1));
% normalization cos(phase) to (-1,1)
phase_vector = normalization(phase_vector, -1, 1);
% calculate phase through arccos()
phase_vector = acos(phase_vector);
% add phase weight of the first mode 
phi = [zeros(number_of_test_images,1) phase_vector];

%% rebuilt phase vector
% define a varibale for complex vectors
complex_vector_N  = zeros(number_of_test_images,number_of_modes);

for i = 1:number_of_test_images
    % read phase weights and generate all possible combinations 
    phi_vectors = phi(i,:).* phase_weight;
    complex_vector_n = zeros(size(phi_vectors,1),number_of_modes);
    % read the ground truth
    ground_truth_i = ground_truth(:,:,i); 
    correlation_n = zeros(size(phi_vectors,1),1);
    
    % reconstruct all possible field distribution
    for j = 1:size(phi_vectors,1)
        complex_vector = amplitude_vector(i,:) .*exp(1i* phi_vectors(j,:)); %                
        
        % 2. generation of complex field distribution
        template = sum(mmf_modes .* shiftdim(complex_vector', -2), 3);    
        
        % calculate the correlation coefficient between reconstrion and ground
        % truth
        correlation = abs(corr2(abs(template), ground_truth_i));
        correlation_n(j) = correlation;
        complex_vector_n(j, :) = complex_vector;
    end       
    % find the right phase weights regarding on the max correlation
    posx = find(correlation_n == max(correlation_n));   
    if numel(posx) > 1
        posx = posx(1);
    end
    complex_vector_N(i,:) = complex_vector_n(posx,:);
end

%% rebuilt the distribution(complex) 
% using function: mmf_build_image()
Image_data_complex = mmf_build_image(number_of_modes, image_size, number_of_test_images, complex_vector_N);
end
