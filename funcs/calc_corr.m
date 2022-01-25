function corr_vals = calc_corr(ground_truth, nn_predictions)
    if size(ground_truth) ~= size(nn_predictions)
        errID = 'myComponent:inputError';
        msgtext = 'Size of gts and preds is not the same';
        ME = MException(errID, msgtext);
        throw(ME)
    end
    
    num_imgs = size(ground_truth, 3);
    corr_vals = zeros(num_imgs, 1);
    for i = 1:num_imgs
        gt = ground_truth(:, :, i);
        nn = nn_predictions(:, :, i);
        corr_vals(i) = abs(corr2(gt, nn));
    end
end

