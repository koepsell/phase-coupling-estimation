% AUTORIGHTS

mex fill_matrix

% test

load testdata/three_phases data K_true

K_fit = fit_model(data);

hval = max(max(K(:),max(K_true(:))));

figure(1)
subplot(121)
imagesc(K_true,[-1 1]*hval)
subplot(122)
imagesc(K_fit,[-1 1]*hval)

K_true
K_fit

K_error = mean(abs(K_true(:)-K_fit(:)));

fprintf('\n mean-absolute-difference=%6.8f ; expect: xxxx',K_error);
