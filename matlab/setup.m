%setup
% 
% 		Description: 
% 		Compiles the C routine (mex file) and tests the
% 		accuracy of the algorithm and the code implementation.
%
% Authors: Charles Cadieu <cadieu@berkeley.edu> and
%          Kilian Koepsell <kilian@berkeley.edu> 
%
% Reference: C. Cadieu and K. Koepsell, A Multivaraite Phase Distribution and its 
%            Estimation, NIPS, 2009 (in submission).

% Copyright (c) 2008 The Regents of the University of California
% All Rights Reserved.

mex fill_matrix.c

% test

load testdata/three_phases data K_true K_python

% fit data with single precision
K_fit = fit_model(single(data));

K_error_single = mean(abs(K_true(:)-K_fit(:)));
code_error_single = mean(abs(K_python(:)-K_fit(:)));

fprintf('\n single precision');
fprintf('\n mean-absolute-difference = %6.8f; expect: 0.01730561',K_error_single);
fprintf('\n difference from python code = %6.8f; expect: 0.0\n',code_error_single);

% fit data with doubleﬂ precision
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

fprintf('\n double precision');
fprintf('\n mean-absolute-difference = %6.8f; expect: 0.01730561',K_error);
fprintf('\n difference from python code = %6.8f; expect: 0.0\n',code_error);
