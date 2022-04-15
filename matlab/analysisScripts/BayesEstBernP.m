function [Pest1,theta1,Pdist1,Pest2,theta2,Pdist2,Pjnt] = BayesEstBernP(D1,theta0,D2,fig,ax)
% Computes the Bayesian estimate of the p parameter(s) of one or two Bernoulli
% distribution(s). When there are 2 distributions in play, this function
% also delivers the joint posterior distribution.
%
% Syntax [Pest1,theta1,Pdist1,Pest2,theta2,Pdist2,Pjnt] = BayesEstBernP(D1,theta0,D2,fig)
%
% the first two arguments (D1 & theta0)  are obligatory. If all one wants
% is the Bayesian estimate of the Bernoulli theta, then
% Syntax  Pest1 =  = BayesEstBernP(D1,theta0)
% D1 is a binary vector
% theta0 is the initial parameter vector for the beta prior on Bern(theta)
% D2 is either another binary vector OR a scalar >0 and <1 specifying a
% comparison value for Bern(theta)
% fig if true causes the function to produce a figure. If a figure is
% desired but there is not D2, set D2 = []. In that case, the figure will
% be the posterior beta distribution. If D2 is not empty, then the figure
% will have a top subplot showing the two poterior distributions and a
% bottom subplot with a contour plot of the joint posterior distribution.
% Pest1 is the mean of the posterior distribution on theta_{Bern1}, that
% is, on the p parameter of the first Bernoulli distribution
% Ditto for Pest2, mutatis mutandis
% theta1 is the updated parameter vector for the beta prior/posterior on
% theta_{Bern1}
% Ditto for theta2, mutatis mutandis
% Pdist1 is a 100x2 array specifying 100 points in the first posterior
% distribution; col 1 is the support vector; col 2 the vector of
% corresponding probabilities
% Ditto for Pdist2, mutatis mutandis
% Pjnt is the 100x100 array giving the joint probabilities.
% The support for Pdist1 (which is the same as the support for Pdist2) and
% Pjnt may be used to compute the Bayesian nDkl

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
if nargin<3
    D2 = [];
    fig = false;
elseif nargin<4
    fig = false;
end
%%
D1 = logical(D1);
theta1 = [sum(D1) sum(~D1)] + theta0; % updating
Pdist1(:,1) = (.005:.01:.995)'; % support for beta posterior(s)
Pdist1(:,2) = betapdf(Pdist1(:,1),theta1(1),theta1(2)); % probability
% densities for the first posterior
Pdist1(:,2) = Pdist1(:,2)/sum(Pdist1(:,2)); % normalizing
Pest1 = sum(Pdist1(:,2).*Pdist1(:,1)); % probability-weighted sum, i.e.,
% the mean of the posterior distribution
%%
if ~isempty(D2)
    Pdist2(:,1) = Pdist1(:,1); % support vector for 2nd beta posterior
    if numel(D2) > 1
        D2 = logical(D2);
        theta2 = [sum(D2) sum(~D2)] + theta0; % updating
        Pdist2(:,2) = betapdf(Pdist2(:,1),theta2(1),theta2(2)); % probability densities
        Pdist2(:,2) = Pdist2(:,2)/sum(Pdist2(:,2)); % normalizing
        Pest2 = sum(Pdist2(:,2).*Pdist2(:,1)); % probability-weighted sum
    elseif D2>0 && D2<1 % D2 specifies a chance probability
        r = find(Pdist2(:,1)>D2,1);
        Pdist2(:,2) = [zeros(r-1,1);1;zeros(100-r,1)]; % puts all the
        % probability mass at the chance (or null) value
        theta2 = [];
        Pest2 = [];
    end
    [PP1,PP2] = ndgrid(Pdist1(:,2),Pdist2(:,2));
    Pjnt = PP1.*PP2;
else
    Pjnt = [];
end
 
%%
if fig
    if isempty(D2)
        figure
        plot(Pdist1(:,1),Pdist1(:,2),'k-','LineWidth',2)
        set(gca,'FontSize',14)
        xlabel('Bernoulli p')
        ylabel('Probability')
        title('Posterior Distribution(s)','FontWeight','normal')
    else
        figure
        subplot(2,1,1)
            plot(Pdist1(:,1),Pdist1(:,2),'k-','LineWidth',2)
            hold on
            plot(Pdist2(:,1),Pdist2(:,2),'k--','LineWidth',2)
            legend('Post1','Post2')
        subplot(2,1,2)
            mx = max(Pjnt(:));
            lvls = mx*[.95 .1 .05 .01];
            contour(Pdist1(:,1),Pdist2(:,1),Pjnt,lvls,'k','LineWidth',1)
    end
end
