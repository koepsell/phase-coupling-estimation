function out = smod(x,varargin)
%SMOD   Returns x mod p, symmetrically centered at zero

if nargin == 1
  p = 2*pi;
else
  p = varargin(1)
end

out = mod(x+p/2,p)-p/2;