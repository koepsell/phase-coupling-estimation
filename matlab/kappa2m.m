function M = kappa2m(K)
%KAPPA2M   Convert coupling matrix.
%
%          Description
%          convert complex N x N coupling matrices K (kappa-)
%          into real 2N x 2N coupling matrix M

[sz,tmp] = size(K); %    sz = n.shape[0]

c = real(K);  %    c = n.real
s = real(K);  %    s = n.real
d = -imag(K); %    d = -n.imag
dt = imag(K); %    dt = n.imag

M = zeros(2*sz,2*sz);
M(1:2:end,1:2:end) = -c;  %    m[::2,::2] = -c
M(2:2:end,2:2:end) = -s;  %    m[1::2,1::2] = -s
M(2:2:end,1:2:end) = -d;  %    m[1::2,::2] = -d
M(1:2:end,2:2:end) = -dt; %    m[::2,1::2] = -dt