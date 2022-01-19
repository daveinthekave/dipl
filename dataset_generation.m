%% Generation of dataset.
% pseudo random mode combination 

%%  set parameters 
number_of_modes = 5;    %option: 3 or 5
number_of_data = 10000;
image_size = 32;    % resolution 32x32
 
%% generation of complex mode weights and label vector - Aufgabe 1

% 1. create random amplitude weights. The weights of amplitude should be normalized.
amp_ws = rand(number_of_modes, number_of_data);
amp_ws = amp_ws ./ vecnorm(amp_ws);

% 2. create random phase amplitude. (Using realtive phase difference)
phs_ws = wrapToPi(rand(number_of_modes, number_of_data) * 2*pi);
rel_phs_ws = phs_ws - phs_ws(1, :);
% what is with mode 1
rel_phs_ws = cos(rel_phs_ws);

% 3. complex mode weights vector

% 4. normalize cos(phase) to (0,1)

% 5. combine amplitude and phase into a label vector (1,2N-1)

% 6. split complex mode weights vector and label vector into Training,
% validation and test set. 

%% create image data - Aufgabe 2 
% use function mmf_build_image()


%% save dataset


