%% Generation of dataset.
% pseudo random mode combination 

%%  set parameters 
number_of_modes = 5;    %option: 3 or 5
number_of_data = 10000;
image_size = 32;    % resolution 32x32
split = floor(0.8 * number_of_data);

%% generation of complex mode weights and label vector - Aufgabe 1

% 1. create random amplitude weights. The weights of amplitude should be normalized.
amp_ws = rand(number_of_modes, number_of_data);
amp_ws = amp_ws ./ vecnorm(amp_ws);

% 2. create random phase amplitude. (Using realtive phase difference)
phs_ws = wrapToPi(rand(number_of_modes, number_of_data) * 2*pi);
rel_phs_ws = phs_ws - phs_ws(1, :);

% 3. complex mode weights vector
cplx_ws = amp_ws .* exp(1i*rel_phs_ws);

% 4. normalize cos(phase) to (0,1)
rel_phs_ws = cos(rel_phs_ws(2:end, :));
norm_phs_ws = normalization(rel_phs_ws, 0, 1); 

% 5. combine amplitude and phase into a label vector (1,2N-1)
labels = [amp_ws; norm_phs_ws];

% 6. split complex mode weights vector and label vector into Training,
% validation and test set. 
train_labels = labels(:, 1:split)';
test_labels = labels(:, split+1:end)';

%% create image data - Aufgabe 2 
% use function mmf_build_image()
cplx_images = mmf_build_image(number_of_modes, image_size, number_of_data, cplx_ws);
images = abs(cplx_images);

%% save dataset
train_imgs = images(:, :, 1, 1:split);
test_imgs = images(:, :, 1, split+1:end);
name = sprintf('%d-modes_%dx%d_%dk.mat', number_of_modes, image_size, image_size, floor(number_of_data/1000));
save(fullfile('data', name), 'train_imgs', 'train_labels', 'test_imgs', 'test_labels')

