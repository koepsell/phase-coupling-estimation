function plot_phase_dist_1d(phases,nbins)
%plot_phase_dist_1d  Plot and fit phase histogram

% put phases in interval [-pi,pi]
phases = smod(phases);

if nargin<2
    nbins = 37;
end

%% fit phase distribution
r = mean(exp(1j*phases(:)));
kappa = fminsearch(@(x)(besseli(1,x)/besseli(0,x)-abs(r))^2,0);
mu = angle(r);
title_txt = sprintf('kappa = %2.2f, mu = %2.2f',kappa,mu);
fprintf([title_txt '\n'])

%%

edges = linspace(-pi,pi,nbins+1);
phi = linspace(-pi,pi,100);
hold off
[n,bins] = histc(phases(:),edges);
bar(edges,n,'histc');
hold on
plot(phi,length(phases)/(4*pi^2*besseli(0,kappa))*exp(kappa*cos(phi-mu)),'r','linewidth',2);
xlim([-pi,pi]);
title(title_txt);

