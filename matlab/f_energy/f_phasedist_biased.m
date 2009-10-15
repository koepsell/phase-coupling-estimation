function F = f_phasedist_biased(theta,M)
x = zeros(1,2*length(theta)+2);
x(1) = 1;
x(3:2:end) = cos(theta);
x(4:2:end) = sin(theta);
F=-.5*x*M*x';
