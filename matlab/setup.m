% AUTORIGHTS

mex fill_matrix.c

% test

load testdata/three_phases data K_true K_python

K_fit = fit_model(data);

hval = max(max(abs(K_fit(:))),max(abs(K_true(:))));

figure(1)
subplot(131)
imagesc(abs(K_true),[-1 1]*hval)
title('True coupling (K\_true)')
axis square off
subplot(132)
imagesc(abs(K_fit),[-1 1]*hval)
title('Estimated coupling, matlab (K\_fit)')
axis square off
subplot(133)
imagesc(abs(K_python),[-1 1]*hval)
title('Estimated coupling, python (K\_python)')
axis square off

K_true
K_fit

K_error = mean(abs(K_true(:)-K_fit(:)));
code_error = mean(abs(K_python(:)-K_fit(:)));

fprintf('\n mean-absolute-difference = %6.8f; expect: 0.01730561',K_error);
fprintf('\n difference from python code = %6.8f; expect: 0.0\n',code_error);
