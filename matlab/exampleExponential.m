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
mu = .2;
data = exprnd(mu,1,100);
data = cumsum(data); % data must be cumulative for the exponential case

%% set prior theta
theta = [.5 0]; % jeffreys for normal

%% update theta based on data
for dataI = 1:length(data)-1
    datum = data(dataI);
    [~,~,theta] = expoupdate(dataI,datum,theta);
end

%% get stats/generate figure
[~,lambdaEstimate] = expoupdate(length(data),data(length(data)),theta,100,true); % final two: number of bins for the posterior and whether to show the summary figure, respectivley
muEstimate = 1/lambdaEstimate;
disp(['mu estimate = ' num2str(muEstimate)])