function theta = normalgamma_update(D, theta0)
% updates the hyperparameters of the normalgamma prior on the mean and
% precsion (=1/var) of the normal distribution.
%      Syntax  theta = normalgamma_update(D, theta0), 
% where theta is the 4-element (hyper)parameter vector of the normalgamma
% distribution, D are data assumed to be drawn from a normal distribution,
% and theta0 gives initial values for that parameter vector. The
% normalgamma is the conjugate prior for the normal distribution when it is
% parameterized by its mean and precision ( = 1/var). One can inject prior
% knowledge about plausible values for the mean and variance of the
% distribution from which one's data come by an astute choice of values for
% theata0. The choice of initial values for these parameters incorporates
% prior knowledge by in effect assuming that one already has some data that 
% indicate the ballpark for the mean and variance. The ball park for a mean 
% and become clear as soon as one sees the first two or three data. Thus, 
% if one is confident that one already knows the ballpark, one can choose
% values for the elements of thet0 that in effect put in the information
% that one has from previous experience, knowledge of what has been
% measured (there are for example lower and upper limits on how fast a rat
% can run a 2 meter runway) and/or from analytic considertions (for example,
% proportions fall on the interval [0 1]. Thus, for example, knowing that
% rats generally run at about 1m/s would lead one to put in a value of 2
% for the first element of theta0, theta0(1). 
%     The cool thing is that you can also indicate your confidence in this
% guesstimate by the value you put in for the 2nd element, theta0(2). This
% is a pseudo sample size. The law of large numbers says that the larger
% the sample size, the more accurate the estimate. A complement to the law
% law of large numbers is the Fisher information, which, roughly speaking,
% tells us that we gain much more information about the parameters of a
% a distriubtion from the first few data we see, then from the 100th datum.
% Thus, as we increase the pseudo sample sizes implicit in our choices of
% values for the elements of theta, we rapidly increase the amount of
% prior information we claim to have about the parameters of the
% distribution from which our data come. In most cases, this counsels us to
% choose low values for these pseudo sample sizes.
%
% SPECIFYING A BALLPARK FOR THE MEAN: Set theta0(1) to your best informed
% guess. An informed guess takes into account prior experience with data of
% a given kind, e.g., how fast rats run, analytic considerations, e.g.,
% running speeds cannot be <0, and obvious physical considerations, e.g.,
% rats cannot run faster than 2m/s. Set theta0(2) = 1 (reasonable
% confidence) or 2 (strong confidence).
%
% SPECIFYING A BALLPARK FOR THE VARIANCE
% this is harder! First choose a value for theta0(3), aka alpha, that
% reflects your confidence in your guesstimate of the variance. Bear in
% mind that theta0(3)is like theta0(2) in that it is a pseudo sample size.
% The implied sample size behind your claimed prior knowledge of the
% variance is 2*theta0(3). Thus, setting theta0(3) = 1 implies a prior
% estimate of the variance based on a sample with n = 2, which is the
% smallest sample size from which it is possible to estimate a variance.
% Thus, 1 is a good cautious value for theta0(3). However, if one has
% strong a priori confidence in the approximate variance and wants to get a
% good estimate of the mean from a very small sample, values as high as,
% say, 10 for theta0(3) might be entertained. You need the value for theta0(3)
%  before you can set the value for theta0(4) because this last value has
%  to be your guesstimate of the sum of squared deviations. This
%  guesstimate is 2*theta0(3)*(variance guesstimate)/2 = theta0(3)*(variance guesstimate),
% because the expected sum of squared deviations is the pseudo sample size
% 2*(theta0(3) times the expected variance).
%
%JEFFREYS PRIOR
% Use theat0 = [0 0 -.5 0];
% The Jeffreys prior is an a minimally informative prior, that is, a prior
% to use when you really have no prior information about what the values of
% the mean and sigma might be or when, as is more often the case, for
% sociological reasons, you wish to pretend that you don't. It has the
% unique property that the parameter estimates you get are invariant under
% change of parameters, which is a mathematical argument in favor of its
% use. What this means is that you get the same parameter estimates whether
% you work with the mean and precision or the mean and the variance
% (assuming the the distribution you use with the precision is converted
% into the corresponding distribution for the variance, using the change of
% parameter formula var = 1/precision!!!). For any other prior, including
% [ 0 0 0 0], this is not true. On the other hand, the Jeffreys prior is
% not consistent with the strong likelihood principle, although I seem to
% have read somewhere that this is only true in the multivatiate normal
% case, not the univariate, or maybe in the mixture case. In any case, its
% use as an uninformative prior in the univariate case is not
% controversial.

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

mu0 = theta0(1);
lambda0 = theta0(2);
alpha0 = theta0(3);
beta0 = theta0(4);

n = length(D);

% The easy ones
lambda = lambda0 + n;
alpha = alpha0 + n/2;

% the hard ones
xbar = mean(D);
mu = ( (lambda0*mu0) + (n*xbar) ) / (lambda0 + n);

% the really hard one
s = var(D,1);
beta = beta0 + 0.5*(n*s + (lambda0*n*(xbar-mu)^2) / (lambda0+n) );

theta = [mu, lambda, alpha, beta];

