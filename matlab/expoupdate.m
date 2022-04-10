function [mxLkLam,mnPostLam,theta,postgam,postmu] = expoupdate(n,T,theta0,nb,fig,ax)
% computes the updated parameters of the gamma posterior on the exponential
% rate parameter, lambda, the posterior distribution on lambda, and the
% posterior distribution on mu = 1/lambda, given n, the # of events
% observed, T, the duration of the observation interval and theta0, the
% initial value of the [alpha beta] (hyper)parameter vector, nb specifies
% the number of bins (points or bars) in the distribution
%
% Syntax [mxLkLam,mnPostLam,theta,postgam,postmu] = expoupdate(n,T,theta0,fig)
%
% maxLkLam is the maximum likelihood estimate of the rate (n/T)
% mnPostLam is the mean of the Bayesian posterior distribution on the rate
% theta is the updated (hyoer)parameter vector; postgam is a 2-col array
% with values for lambda in the first column and corresponding p values in
% the second column; postmu is the same distribution after change of
% variable (from lambda  to mu)

% MIT License
% 
% Copyright (c) 2022 bendecorte
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

if nargin<4
    nb = 100;
end
if nargin<5
    fig = false;
end
theta(1) = theta0(1) + n; % shape parameter
theta(2) = theta0(2) + T; % beta parameter
mxLkLam = n/T; % maximum likelihood estimate of the rate
B = 1/theta(2); % Matlab's scale parameter (theta in Wikipedia notation)
lam = linspace(gaminv(.001,theta(1),B),gaminv(.999,theta(1),B),nb)'; % support
% vector (values for the rate)
dlam = lam(2)-lam(1); % increment (granularity) of support
% 100-element lambda vector
P = gamcdf(lam,theta(1),B); % cum prob vs lam
pm = diff(P); % prob mass in each of 99 successive intervals
pm = pm/sum(pm); % normalizing
lam = lam(2:end)-dlam/2; % centered values of lambda
postgam = [lam pm]; % the posterior gamma distribution on the exponential's
% rate parameter
postmu = [1./lam pm]; % posterior distribution after change of variable
% (from the rate parameter to the mean parameter of the exponential
% (=1/rate)
mnPostLam = sum(pm.*lam); % mean of the Bayesian posterior on the rate
%% plotting posteriors
if fig
    if exist('ax','var')
        bar(ax,postgam(:,1),postgam(:,2),1,'FaceColor',[.5 .5 .5],'EdgeColor',[.5 .5 .5])
        set(gca,'FontSize',14)
        xlabel('\it\lambda','FontSize',18)
        ylabel('probability','FontSize',18)
    else
        figure
        subplot(2,1,1)
            bar(postgam(:,1),postgam(:,2),1,'FaceColor',[.5 .5 .5],...
                'EdgeColor',[.5 .5 .5])
            xlabel('\it\lambda','FontSize',18)
            ylabel('probability')
        subplot(2,1,2)
            plot(postmu(:,1),postmu(:,2),'k*')
            xlabel('\it\mu','FontSize',18)
            ylabel('probability')
    end
end