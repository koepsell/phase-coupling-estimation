function K = fit_model(p)
%fit_model  Multivariate Phase Distribution Estimation 
% 
% 			Description: 
% 			K = FIT_MODEL(P,VARS) estimates the parameters of a multivariate 
%			phase distribution using the distribution and method described in [1].
%
%			Inputs:
%			P = d-by-n matrix of phase measurements where d is the dimensionality, 
%				and n is the number of data points.
%
%			Outputs:
%			K = the estimated K for the multivariate phase distribution
%
%		[1] C. Cadieu and K. Koepsell, A Multivaraite Phase Distribution and its 
%			Estimation, NIPS, 2009 (in submission).
%			
% AUTORIGHTS

% prepare phaseinput, matrices, and parameters
[d,nz] = size(p);

z = reshape([ exp(1j*p); exp(-1j*p)],[2,d,nz]);

nij   = d^2 - d; % number of coupling terms
na    = 4*d^3-d^2+6; % upper bound for number of elements in sparse matrix

% adata = complex(zeros(1,na),zeros(1,na));
% arow  = zeros(1,na,'int8');
% acol  = zeros(1,na,'int8');
% b     = complex(zeros(1,nij),zeros(1,nij));

% call fill_matrix to create linear set of equations (modifies data in place)
tic
[adata,arow,acol,b] = fill_matrix(z,nij,na);
toc

% solve linear system of equations
a = sparse(arow,acol,adata,nij,nij); % uses nzmax = length(adata).
kij = linsolve(a,b);

% prepare result for return
K = zeros(d,d);
K(~diag(ones(d,1))) = -kij;
K = K';
