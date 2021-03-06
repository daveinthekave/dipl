function [image_data] = mmf_build_image(number_of_modes, image_size, number_of_data, complex_weights_vector)
%% load complex mode distribution
% load the complex distrbutions
if number_of_modes == 5
    load('mmf_5modes_32.mat')
else
    errID = 'myComponent:inputError';
    msgtext = sprintf('No file for %d modes', number_of_modes);
    ME = MException(errID,msgtext);
    throw(ME);
end

%% create images
% define a variable for Image data with dimension (image size, image size, 1, n)
fprintf("Start to generate the mode distribution......\n");
image_data = zeros(image_size, image_size, 1, number_of_data);
for index_number = 1:number_of_data
    % 1. define a variable for single image with resolution (image size,image size)
    weights = complex_weights_vector(index_number, :);
    
    % 2. generation of complex field distribution 
    cplx_field = sum(mmf_5modes_32 .* shiftdim(rot90(weights, 3), -2), 3);
    
    % 3. abstract Amplitude distribution    
    % 4. normalization the amplitude distribution to (0,1)
    %    using normalization(image, minValue, maxValue)
    norm_amp = normalization(abs(cplx_field), 0, 1);
    
    norm_cplx_field = norm_amp .* exp(1i*angle(cplx_field));
    image_data(:, :, 1, index_number) = norm_cplx_field;   
    
end
fprintf("The image data has been generated.\n");

end

