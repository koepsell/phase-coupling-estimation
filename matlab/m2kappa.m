function K = m2kappa(M)
%M2KAPPA   Convert coupling matrix.
%
%          Description
%          convert real 2N x 2N coupling matrices M
%          into complex N x N coupling matrix K (kappa-)

c = -M(1:2:end,1:2:end); %    c = -m[::2,::2]
s = -M(2:2:end,2:2:end); %    s = -m[1::2,1::2]
d = -M(2:2:end,1:2:end); %    d = -m[1::2,::2]
% P = -M(1:2:end,1:2:end); %    p = .5*(c-s) + .5j*(d.T+d)
K = .5*(c+s) + .5j*(transpose(d)-d); %    n = .5*(c+s) + .5j*(d.T-d)

