function K = fit_model(p,vars)

% prepare phaseinput, and parameters
[d,nz] = size(p);

z = reshape([ exp(1j*p); exp(-1j*p)],[2,d,nz]);

nij = d^2 - d; % number of coupling terms
na  = 4*d^3-d^2+6; % upper bound for number of elements in sparse matrix
adata = complex(zeros(1,na),zeros(1,na));
b     = complex(zeros(1,na),zeros(1,na));
arow  = zeros(1,na,'int8');
acol  = zeros(1,na,'int8');

% call fill_matrix to create linear set of equations
tic
fill_matrix(z,adata,arow,acol,b);
toc

% solve linear system of equations
a = sparse(arow,acol,adata,nij,nij); % uses nzmax = length(adata).
kij = linsolve(a,b);

% prepare result for return
K = zeros(d,d);
K(~diag(ones(d,1))) = -kij;
K = K';