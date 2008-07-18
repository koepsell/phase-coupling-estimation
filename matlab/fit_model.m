function K = fit_model(p)
%fit_model  Multivariate Phase Distribution Estimation 
% 
% 			Description: 
% 			K = FIT_MODEL(P,VARS) estimates the parameters of a multivariate 
%			phase distribution using the distribution and method described in [1].
%
%			Inputs:
%			P = d-by-n matrix of phase measurements where d is the dimensionality, 
%				and n is the number of data points.
%
%			Outputs:
%			K = the estimated K for the multivariate phase distribution
%
%		[1] C. Cadieu and K. Koepsell, A Multivaraite Phase Distribution and its 
%			Estimation, NIPS, 2009 (in submission).
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

% prepare phaseinput, matrices, and parameters
[d,nz] = size(p);

nij   = d^2 - d; % number of coupling terms
na    = 4*d^3-10*d^2+6*d; % upper bound for number of elements in sparse matrix

% adata = complex(zeros(1,na),zeros(1,na));
% arow  = zeros(1,na,'int8');
% acol  = zeros(1,na,'int8');
% b     = complex(zeros(1,nij),zeros(1,nij));


% call fill_matrix to create linear set of equations (modifies data in place)
tic
[a,b] = fill_matrix(exp(1j*p),nij,na);

% solve linear system of equations
kij = a\b;
toc

% prepare result for return
K = zeros(d,d);
K(~diag(ones(d,1))) = -kij;
K = K';
