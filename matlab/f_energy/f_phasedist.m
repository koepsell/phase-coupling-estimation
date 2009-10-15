function F = f_phasedist(theta,M)
x = zeros(1,2*length(theta));
x(1:2:end) = cos(theta);
x(2:2:end) = sin(theta);
F=-.5*x*M*x';