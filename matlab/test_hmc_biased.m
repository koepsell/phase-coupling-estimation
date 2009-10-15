% Test script for hybrid montecarlo sampling
% 
% In order to use hybrid montecarlo sampling,
% the functions hmc2.m and hmc2_opt.m are needed.
% Both files are distributed separately here:
% http://www.lce.hut.fi/research/mm/mcmcstuff/mcmcstuff.tar.gz

%% setup

addpath('f_energy');

%% load data

% load('testdata/three_phases.mat','K_true','data');
% sz = size(K_true,1);

%%

dim = 3;
K_true = randn(dim+1,dim+1)+1j*randn(dim+1,dim+1);
K_true(logical(eye(dim+1))) = 0;
K_true = .5*(K_true+conj(transpose(K_true)));
sz = size(K_true,1)-1;
% K_true(2:end,2:end) = 0;

%% convert coupling from complex 3x3 to real 6x6 matrix

M = kappa2m(K_true);

%% some settings

nsamples = 10^3;
burnin = 10^3;
lf_steps = 50;
step_sz = .15;
persistence = 0;

%% sample some data

opts = hmc2_opt();
opts.nsamples = nsamples;
opts.nomit = burnin;
opts.steps = lf_steps;
opts.stepadj = step_sz;
opts.persistence = persistence;

[samps, E, diagn] = hmc2('f_phasedist_biased',zeros(sz,1),opts,'g_phasedist_biased',M);
[nsamples,dim] = size(samps);
data = zeros(dim+1,nsamples);
data(2:end,1:end) = smod(transpose(samps));

%% fit data with double precision

K_fit = fit_model(data);

%% plot results

hval = max(max(abs(K_fit(:))),max(abs(K_true(:))));

figure(1)
subplot(121)
imagesc(abs(K_true),[-1 1]*hval)
title('True coupling (K\_true)')
axis square off
subplot(122)
imagesc(abs(K_fit),[-1 1]*hval)
title('Estimated coupling, matlab (K\_fit)')
axis square off

K_true
K_fit

K_error = mean(abs(K_true(:)-K_fit(:)));
fprintf('\n double precision');
fprintf('\n mean-absolute-difference = %6.8f; expect: ~ 0.07\n',K_error);

