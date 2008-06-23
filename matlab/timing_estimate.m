function [est_time] = timing_estimate(dim_est,samps_est)

%timing_estimate 
%           Estimate time required for a given computation
% 
% 			Description: 
% 			EST_TIME = TIMING_ESTIMATE(DIM,SAMPS) estimates the time required to compute
%           the model estimate for a given dimensionality (DIM) and samples (SAMPS).
%
%			Outputs:
%			EST_TIME = estimated time (in seconds)
%

% Copyright (c) 2008 The Regents of the University of California
% All Rights Reserved.

% Created by Charles Cadieu and Kilian Koepsell (UC Berkeley)

% Permission to use, copy, modify, and distribute this software and its
% documentation for educational, research and non-profit purposes,
% without fee, and without a written agreement is hereby granted,
% provided that the above copyright notice, this paragraph and the
% following three paragraphs appear in all copies.

% This software program and documentation are copyrighted by The Regents
% of the University of California. The software program and
% documentation are supplied "as is", without any accompanying services
% from The Regents. The Regents does not warrant that the operation of
% the program will be uninterrupted or error-free. The end-user
% understands that the program was developed for research purposes and
% is advised not to rely exclusively on the program for any reason.

% This software embodies a method for which a patent is pending.

% IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY
% FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
% INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND
% ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN
% ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE UNIVERSITY OF
% CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
% LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
% A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS"
% BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE
% MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

samps = 4000;
dims = [2 4 6 8 10 12 14 16 20 24];
reps = 10;

t_vals = zeros(1,length(dims));
t_reps = zeros(1,reps);

for d = 1:length(dims)
    p = 2*pi*rand(dims(d),samps);
    for i = 1:reps
        tic;
        fit_model(p);
        t_reps(i) = toc;
    end
    t_vals(d) = mean(t_reps);%mean(t_reps(t_reps<median(t_reps)));
end

coef = polyfit(dims,t_vals,3);

est_time = polyval(coef,dim_est)*samps_est/samps;
if est_time < 60
    sprintf('\nEstimated time: %0.5g seconds',est_time)
elseif est_time < 3600
    sprintf('\nEstimated time: %0.5g minutes',est_time/60)    
else
    sprintf('\nEstimated time: %0.2g hours',est_time/3600)
end

figure(2)
clf
ds = 1:100;
plot(ds,polyval(coef,ds)*100000/samps/3600)
hold on
plot(dims,t_vals*100000/samps/3600,'ro')
xlabel('dimensionality')
ylabel('time (hours)')
title('time for 100,000 samples')
