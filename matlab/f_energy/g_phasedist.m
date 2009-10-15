function G = g_phasedist(theta,M)
x = zeros(1,2*length(theta));
x(1:2:end) = cos(theta);
x(2:2:end) = sin(theta);
xdot = zeros(2*length(theta),length(theta));
xdot(1:2:end,:) = diag(-sin(theta)); 
xdot(2:2:end,:) = diag(cos(theta)); 
G=-x*M*xdot;