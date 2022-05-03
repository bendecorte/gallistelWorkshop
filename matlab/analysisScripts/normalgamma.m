function P = normalgamma(x,tau,theta,fig,ax)
% computes the normalgamma probability density function on the mu and tau
% (precision = 1/var) parameters of the normal distribution, given a 
% vector of possible values for the mean (x) a vector of possible values
% for tau and the 4-element parameter vector, theta, of the normalgamma 
% distribution (the hyperparameters). This parameter vector is computed from the
% normalgamma_update function when given data and initial values for theta.
% The normalgamma is the conjugate prior for the normmal distribution. The
% Jeffreys prior on the normal distribution assumes an initival value for
% this vector of [0 0 -.5 0]. When x and tau are vectors of length n_x and
% n_tau, p is an n_tau by n_x array. Thus, the marginal distribution on the
% normal mean is the vector obtained by summing across the columns of p and
% the marginal distribution on tau is obtained by summing across the rows.
% The asterisk in the contour plot of the posterior marks the maximum
% likelihood (summit); the contours are 0.5, 0.1, 0.05 and 0.01 of the
% summit from innermost to outermost, respectively.
% the

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

if nargin<4
    fig = false;
    ax = false;
end
if nargin<5
    ax=false;
end 
mu     = theta(1);
lambda = theta(2); % aka nu (in e.g. NormGamPost)
alpha  = theta(3);
beta   = theta(4);

[X,TAU] = ndgrid(x,tau); % arrays that cross all valeus of the x vector with
% all values of the tau vector
  
P = gampdf(TAU, alpha, 1/beta) .* normpdf(X, mu, 1./sqrt(lambda.*TAU));


if islogical(fig) && fig==true% create figure
    figure;set(gcf,'OuterPosition',[240.00 527.00 560.00 493.00])
    subplot(2,2,1)
        plot(x,sum(P,2))
        xlabel('\mu','FontSize',18)
        ylabel('Probability Density','FontSize',12)
        title('Marginal Distributions','FontWeight','normal','FontSize',14)
    subplot(2,2,3)
        plot(tau,sum(P))
        xlabel('\tau  (1/\sigma^2)','FontSize',18)
        ylabel('Probability Density','FontSize',12)
    subplot(2,2,2)
        mesh(tau,x,P)
        ylabel('\mu','FontSize',18)
        xlabel('\tau  (1/\sigma^2)','FontSize',18)
        zlabel('Prob Density','FontSize',12)
        title('Posterior NormalGamma','FontWeight','normal','FontSize',14)
    subplot(2,2,4)
        sz = size(P);
        [M,I] = max(P(:)); % peak probability
        [r,c] = ind2sub(sz,I);
        levels = M*[.01 .05 .1 .5];
        contour(X,TAU,P,levels,'k')
        hold on
        plot(x(r),tau(c),'k*')
        xlabel('\mu','Fontsize',18)
        ylabel('\tau = (1/\sigma^2)','Fontsize',18)
        th = round(theta,2);
        str = ['\theta = [' num2str(th(1)) ' ' num2str(th(2)) ' ' ...
            num2str(th(3)) ' ' num2str(th(4)) ']'];
        title(str,'FontWeight','normal','FontSize',14)
end

if ishandle(fig) && ishandle(ax)
    figure(fig)
    sz = size(P);
    [M,I] = max(P(:)); % peak probability
    [r,c] = ind2sub(sz,I);
    levels = M*[.01 .05 .1 .5];
    contour(ax,X,TAU,P,levels,'k')
    hold on
    plot(x(r),tau(c),'k*')
    xlabel('\mu','Fontsize',18)
    ylabel('\tau (1/\sigma^2)','Fontsize',18)
    th = round(theta,2);
    str = ['\theta = [' num2str(th(1)) ' ' num2str(th(2)) ' ' ...
        num2str(th(3)) ' ' num2str(th(4)) ']'];
    title(str,'FontWeight','normal','FontSize',14)
end
    