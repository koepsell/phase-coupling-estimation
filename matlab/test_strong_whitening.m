
%% setup

addpath('f_energy');
mex -largeArrayDims fourth_corr.c

%%

dim = 3;
K_true = randn(dim,dim)+1j*randn(dim,dim);
K_true(logical(eye(dim))) = 0;
K_true = .5*(K_true+conj(transpose(K_true)));
sz = size(K_true,1);

%% convert coupling from complex 3x3 to real 6x6 matrix

M = kappa2m(K_true);

%% some settings

nsamples = 10^4;
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

[samps, E, diagn] = hmc2('f_phasedist',zeros(sz,1),opts,'g_phasedist',M);
p = smod(transpose(samps));

%% strong uncorrelating transform

%
% x' is hermitean conjugate
%
x = exp(1j*p);
C = inv(sqrtm(x*x'/length(x)));

% conventional whitening:
y = C*x;
abs(y*y'/length(y))

pconv = y*transpose(y)/length(y);
abs(pconv)

%% takagi diagonalization
addpath('/data/matlab');

A = pconv;
n = size(pconv,1);

% generate complex symmetric matrix with singular values sv
% sv = rand(n,1);
% sv = sort(-sv); sv = -sv;
% A = csgen(sv);

nSteps = 10;
% Lanczos tridiagonalization using partial orthogonalization
% [a,b,Q1,nSteps] = LanPO(A,rand(n,1),nSteps);
%
% Lanczos tridiagonalization using modified partial orthogonalization
[a,b,Q1,nSteps] = LanMPO(A,rand(n,1),nSteps);
%
% Lanczos tridiagonalization using modified partial orthogonalization
% and restart
% [a,b,Q1,nSteps,nVec] = LanMPOR(A,rand(n,1),nSteps);

% get number of iterations to run
fprintf('Number of iterations actually run: %d\n', nSteps);

% calculate and report errors
tmp = norm(Q1'*Q1 - eye(nSteps), 'fro')/(nSteps*nSteps);
fprintf('Error in orthogonality in tridiagonalization: %E\n', tmp);
tmp = norm(Q1'*A*conj(Q1) - (diag(a)+diag(b,1)+diag(b,-1)), 'fro');
tmp = tmp/(nSteps*nSteps);
fprintf('Error in tridiagonalization: %E\n', tmp);

% [s,Q2] = cstsvdt(a, b); % twisted SVD

[s,Q2] = CSSVD(a, b);		% pure QR 

Q = Q1*Q2;

% check results
tmp = norm(Q'*Q - eye(nSteps), 'fro')/(nSteps*nSteps);
fprintf('Error in orthogonality: %E\n', tmp);

% calculate and report errors
if nSteps==n
    % fprintf('Error in singular values: %E\n', norm(s - sv)/n);

    tmp = norm(A - Q*diag(s)*conj(Q'), 'fro')/(n*n);
    fprintf('Error in Takagi factorization: %E\n', tmp);
end

z = Q'*y;

% test

% covariance
abs(y*y'/length(y))
abs(z*z'/length(z))

% pseudo covariance
abs(z*transpose(z)/length(z))

%% plot data

figure()
plot_phase_dist_nd(angle(x));

figure()
plot_phase_dist_nd(angle(z));

%% fit model

K_fit = fit_model(smod(transpose(samps)))
K_dec = fit_model(angle(z))

