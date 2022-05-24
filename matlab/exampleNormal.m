% minimal illustration of using analysis on normally-distributed data

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

%% generate data
mu = 2;
sigma = .2;
data = normrnd(mu,sigma,1,50);

%% set prior theta
theta = [0 0 -.5 0]; % jeffreys for normal

%% update theta based on data
for dataI = 1:length(data)
    datum = data(dataI);
    theta = normalgamma_update(datum,theta);
end

%% get stats/generate figure
plausibleMus  = mu + linspace(-4*sigma,4*sigma,200); 
plausibleTaus = linspace(.001,(1/(sigma^2))*4.5,200);
[posteriorPDF,muEstimate,tauEstimate] = normalgamma(plausibleMus,plausibleTaus,theta,true);
sigmaEstimate = 1/sqrt(tauEstimate);

disp(['mu estimate = ' num2str(muEstimate)])
disp(['sigma estimate = ' num2str(sigmaEstimate)])




