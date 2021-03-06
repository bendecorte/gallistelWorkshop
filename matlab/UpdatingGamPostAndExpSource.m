% script m-file UpdatingGamPostAndExpSource.m

%  MIT License
%  Copyright (c) 2022 bendecorte
%  Permission is hereby granted, free of charge, to any person obtaining a copy
%  of this software and associated documentation files (the "Software"), to deal
%  in the Software without restriction, including without limitation the rights
%  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
%  copies of the Software, and to permit persons to whom the Software is
%  furnished to do so, subject to the following conditions:
%  The above copyright notice and this permission notice shall be included in all
%  copies or substantial portions of the Software.
%  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
%  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
%  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
%  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
%  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
%  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
%  SOFTWARE.

addpath(genpath(pwd),'-begin') % top of search path in case of conflicting dependencies
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
if isOctave; pkg load statistics; end

%% fig setup
set(0,'defaultAxesFontSize',24)
close
figure; set(gcf,'units','normalized','OuterPosition',[0.1961 0.0250 0.2858 0.7380])
ax(1) = subplot(2,1,1); set(gca,'FontSize',18)
ax(2) = subplot(2,1,2); set(gca,'FontSize',18)

%% analysis
t = linspace(0,50); % support for the Source distribuution
theta0 = [.5 0]; % Jeffreys prior (initial value of gammapdf pameter vector)
D = exprnd(10,50,1); % 50 draws from an exponential distribution with a mean
% interval between events of 10 s

for d = 1:50 % stepping through the data
    [~,mnPostLam,theta,postgam] = expoupdate(d,sum(D(1:d)),theta0);
    % mnPostLam = estimate of exponential lambda; theta = updated parameter 
    % vector of gammapdf posterior; postgam = updated posterior
    % gamma distribution--but with unnomrmalized probability density vector
    % hence likelihoods not probability densities

    hold(ax(1),'off'); hold(ax(2),'off');
    plot(ax(1),postgam(:,1),postgam(:,2),'k-','LineWidth',2) % plotting posterior
    plot(ax(2),t,exppdf(t,1/mnPostLam),'k-','LineWidth',2) % plotting Source
    
    hold(ax(1),'on')
    plot(ax(1),[.1 .1],ylim(ax(1)),'k--','LineWidth',1) % true value of lambda
    d100 = .01*max(postgam(:,2)); % 100 times less likely than the max likelihood
    plot(ax(1),xlim,[d100 d100],'k:','LineWidth',2)

    title(ax(1),'Posterior Likelihood Function','FontWeight','normal','FontSize',24)
    ylabel(ax(1),'Likelihood','FontSize',24)
    xlabel(ax(1),'Possible Values for \lambda','FontSize',24)
    xlim(ax(1),[0 1])
    ylim(ax(1),[0 .03])
    title(ax(2),'Estimated Source Distribution','FontWeight','normal','FontSize',24)
    ylabel(ax(2),'Probability Density','FontSize',24)
    xlabel(ax(2),'d = Duration (s)','FontSize',24)

    text(ax(1),.15,max(ylim(ax(1)))*.85,...
        ['p(\lambda|D) \propto gampdf(\lambda=0:.01:1, n=' num2str(d) ', D=' num2str(sum(D(1:d)),3) ')'],...
        'FontSize',24) % expression for the posterior distribution
    text(ax(2),10,max(ylim(ax(2)))*.85,'p^{\prime} = \lambdae^{-\lambdad}','FontSize',36) %

    pause
end
