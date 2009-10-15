function plot_phase_dist_nd(phases,nbins)
%plot_phase_dist_nd  Plot and fit phase histogram

if nargin<2
    nbins = 37;
end

% prepare phaseinput, matrices, and parameters
[d,nz] = size(phases);

if nz ~= length(phases)
    error('wrong data size: data samples have to be in matrix columns');
end

clf()
i=0;
hmax = 4*nz/nbins;
for row = 1:d
    for col = 1:d
        i=i+1;
        subplot(d,d,i)
        if col == row
            fprintf('\nphi_%d: ',row)
            plot_phase_dist_1d(phases(row,:),nbins)
            ylim([0,hmax])
            title(sprintf('\\phi_%d',row))
        elseif col > row
            fprintf('phi_%d + phi_%d: ',row,col)            
            plot_phase_dist_1d(phases(row,:)+phases(col,:),nbins)
            ylim([0,hmax])
            title(sprintf('\\phi_%d + \\phi_%d',row,col))            
        else
            fprintf('phi_%d - phi_%d: ',row,col)
            plot_phase_dist_1d(phases(row,:)-phases(col,:),nbins)
            ylim([0,hmax])
            title(sprintf('\\phi_%d - \\phi_%d',row,col))            
        end
    end
end