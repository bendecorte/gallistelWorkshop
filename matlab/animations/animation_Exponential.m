% Example animation for estimating an exponential distribution
% Written for opengl rendering. May be sluggish otherwise. 
% All code by BJD. Haven't cleaned this up too much, but this is primarily
% intended as a visual resource. See other documentation/scripts for
% implementation instructions. 

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

%% startup
close all; clear all
% addpath(genpath(pwd),'-begin') % add directory contents to top of the search path (won't save it, so you can clear everything by restarting matlab). 
showUpdateTime = false; % times each iteration and display while running. 

%% general parameters
numDataPoints = 100;
updateSpeedSecs = .25; % time between draws/updates

%% source parameters
mu = .2; 
dataPoints = exprnd(mu,1,numDataPoints);

%% prior parameters
theta = [.5 0]; % JEFFREYS prior

%% setup figure / make axes
figAnimation = makeAnimationFigure();
[axSource,axPosterior,axEstimateTracking] = makeAnimationAxes();

%% source axis (comes second because it needs estiamtes from the posterior setup above)
maxX = expinv(.99,mu); % cut the x-axis at the point where 99% of the density will fall below 
sourceXValues = linspace(0,maxX,100);
setupSourceAxis(axSource,sourceXValues)
[sourceLine,sourceEstimateLine,dataLine] = makeSourceAxisLines(axSource,sourceXValues,mu);
axSource.YLim = [0 max([sourceLine.YData sourceEstimateLine.YData])*1.05];
sourceAxisLegend = makeSourceAxisLegend(axSource,sourceLine,sourceEstimateLine,dataLine); % for flexibility, actually a cell array of text-objects, rather than typical axis legend

%% prior/posterior axis setup
plausibleLambdas = linspace(.001,3*(mu),100); % note that we can't use 0 because of gamma distribution 
setupPosteriorAxis(axPosterior,plausibleLambdas)
[posteriorLine,plausibleLamdas] = makePosteriorLine(axPosterior,theta,plausibleLambdas); % note that lamdas get slight correction due to the way the pdf is computed (diff of cdf). 

maxLikeLine = makeMaxLikeLine(axPosterior);
makePosteriorAxisLegend(axPosterior,posteriorLine,maxLikeLine);

%% parameter history axis setup
setupEstimateTrackingAxis(axEstimateTracking,dataPoints,plausibleLambdas);
[trueLambdaLine] = makeTrueParameterLine(axEstimateTracking,mu);
[lambdaPointArray] = makeEstimateScatterArrays(axEstimateTracking,dataPoints); % basically, preallocating scatter plots for each iteration
estimateTrackingAxisLegend = makeEstimateTrackingAxisLegend(axEstimateTracking); % I take no shame in how long that function name is. 

for dataI = 1:length(dataPoints)
    updateTimer = tic;
    %% update parameter vector (for first iteration, just draw based on starting values)
    if dataI>0
        currentDataPoint = dataPoints(dataI);
        [maxLikeLambda,meanPosteriorLambda,theta,posteriorGamma,posteriorMu] = expoUpdate_animation(dataI,sum(dataPoints(1:dataI)),theta);
        dataLine.XData = [currentDataPoint currentDataPoint];
        if dataI==1; dataLine.Visible = 'on'; sourceEstimateLine.Visible = 'on'; maxLikeLine.Visible = 'on'; end % show line when first real value is drawn
    end

    %% update the posterior axis plot objects
    % get the posterior at all values of interest
    posteriorPDF = posteriorMu(:,2); 
    % PDF 
    posteriorLine.XData = posteriorMu(:,1);
    posteriorLine.YData = posteriorPDF;
    % maximum likelihood
    maxLikeLine.XData = [1/meanPosteriorLambda 1/meanPosteriorLambda];
    maxLikeLine.YData = axPosterior.YLim;
    axPosterior.YLim = [0 max(posteriorLine.YData)*1.2];
    
    %% draw the current estimate of the source distribution, given the parameters from the updated posterior
    currentLambdaEstimate = meanPosteriorLambda;
    currentSourceEstimatePDF = exppdf(sourceXValues,1/currentLambdaEstimate);
    sourceEstimateLine.YData = currentSourceEstimatePDF;
    axSource.YLim = [0 max([sourceLine.YData sourceEstimateLine.YData])*1.2];
    
    %% plot estimate on bottom panel
    lambdaPointArray(dataI+1).YData = 1/currentLambdaEstimate;
    lambdaPointArray(dataI+1).Visible = 'on'; % when preallocated, all points are set to be invisible
    
    %% show current theta on the history axis (best place for this?)
    setEstimateTrackingAxisTitle(axEstimateTracking,round(theta,2))
    
    %% render and pause
    drawnow expose
    pause(updateSpeedSecs)
    
    %% check rendering time
    if showUpdateTime; disp(['update timer: ' num2str(toc(updateTimer))]); end
end






%% supporting functions
function [mxLkLam,mnPostLam,theta,postgam,postmu] = expoUpdate_animation(n,T,theta0,nb)
    if nargin<4; nb = 100; end
    theta(1) = theta0(1) + n; % shape parameter
    theta(2) = theta0(2) + T; % beta parameter
    mxLkLam = n/T; % maximum likelihood estimate of the rate
    B = 1/theta(2); % Matlab's scale parameter (theta in Wikipedia notation)
    lam = linspace(gaminv(.001,theta(1),B),gaminv(.999,theta(1),B),nb)'; % support vector (values for the rate)
    dlam = lam(2)-lam(1); % increment (granularity) of support 100-element lambda vector
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
end

function [figAnimation] = makeAnimationFigure()
    figAnimation = figure;
    figAnimation.Color = [1 1 1];
    figAnimation.Units = 'normalized';
    figAnimation.Position = [0.2599 0.0796 0.4505 0.8000]; % chosen manually
    try
        figAnimation.Renderer = 'opengl';
    catch
        Warning('Warning: Could not set renderer to opengl. Graphics may be slow.')
    end
end

function [axSource,axPosterior,axEstimateTracking] = makeAnimationAxes()
    numAxRows = 3; numAxCols = 1;
    axSource = subplot(numAxRows,numAxCols,1);
    axPosterior = subplot(numAxRows,numAxCols,2);
    axEstimateTracking = subplot(numAxRows,numAxCols,3);
    axesToEdit = [axSource axPosterior axEstimateTracking];
    for axI = 1:length(axesToEdit)
        currentAx = axesToEdit(axI);
        hold(currentAx,'on')
        currentAx.FontName = 'Arial';
        currentAx.FontSize = 14;
        currentAx.FontWeight = 'Bold';
        currentAx.XLabel.FontSize = 14; currentAx.YLabel.FontSize = 14;
    end
end

function setupSourceAxis(axSource,sourceXValues)
    axSource.XLim = [min(sourceXValues) max(sourceXValues)];
    axSource.XLabel.String = 'inter-response intervals (secs)'; axSource.YLabel.String = 'density';
    axSource.Title.String = 'Source (true vs. estimate)';
end

function [sourceLine,sourceEstimateLine,dataLine] = makeSourceAxisLines(axSource,sourceXValues,lambda)
    % true source 
    sourcePDF = exppdf(sourceXValues,lambda);
    sourceLine = plot(axSource,sourceXValues,sourcePDF);
    sourceLine.Color = [0 0 0];
    sourceLine.LineWidth = 2; 
    sourceLine.LineStyle = '--';
    
    % estimate of true source from the analysis
    sourceEstimateLine = plot(axSource,sourceXValues,zeros(size(sourceXValues)));
    sourceEstimateLine.Color = [0 1 0];
    sourceEstimateLine.LineWidth = sourceLine.LineWidth;
    sourceEstimateLine.Visible = 'off'; % just preallocating, so don't show yet.
    
    % used to show the value of the datum drawn at each iteration
    dataLine = plot(axSource,[lambda lambda],axSource.YLim);
    dataLine.YData(2) = dataLine.YData(2)*.65;
    dataLine.LineWidth = 4; 
    dataLine.Color = [.7 .7 .7];
    dataLine.Visible = 'off'; 
end

function sourceAxisLegend = makeSourceAxisLegend(axSource,sourceLine,sourceEstimateLine,dataLine)
    fontSize = 14;
    textStrings = {'true','estimate','new datum'};
    textColors = [sourceLine.Color; sourceEstimateLine.Color;dataLine.Color];
    ySpacing = diff(axSource.YLim) * .2; % just a fraction the y-axis span (i.e., data-units) to keep it normalized.
    legendX = axSource.XLim(2) * .98;
    legendY = axSource.YLim(2) - (.5*ySpacing);
    for entryI = 1:length(textStrings)
       currentEntry = text(axSource, legendX, legendY,textStrings{entryI}); 
       currentEntry.Units = 'normalized'; % in case user sets y-axis to update based on estimated pdf
       currentEntry.HorizontalAlignment = 'right';
       currentEntry.Color = textColors(entryI,:);
       currentEntry.FontName = 'Arial'; currentEntry.FontSize = fontSize;
       sourceAxisLegend{entryI} = currentEntry;
       legendY = legendY - ySpacing;
    end
end

function setupPosteriorAxis(axPosterior,plausibleLambdas)
    axPosterior.XLim = [min(plausibleLambdas) max(plausibleLambdas)];
    axPosterior.XLabel.String = '\lambda (responses / sec)'; 
    %axPosterior.XLabel.FontSize  = 18;
    axPosterior.YLabel.String = 'density';
    axPosterior.YLabel.FontSize = axPosterior.XLabel.FontSize;   
end

function [posteriorLine,plausibleLambdas] = makePosteriorLine(axPosterior,theta,plausibleLambdas)
    % unpack theta for clarity
    alpha = theta(1);
    beta = theta(2);
    scale = 1/beta;
    % compute pdf
    deltaLambda = plausibleLambdas(2)-plausibleLambdas(1); % increment (granularity) of support 100-element lambda vector
    cumulativeDensity = gamcdf(plausibleLambdas,theta(1),scale); % cum density. to do: was expecting matlab to return inf at 0 or negative values. Returns 0 in reality. Is that right?
    posteriorPDF = diff(cumulativeDensity); % prob mass in each of successive intervals
    posteriorPDF = posteriorPDF/sum(posteriorPDF); % normalizing
    plausibleLambdas = plausibleLambdas(2:end) - (deltaLambda/2); % center values of lambda to account for taking derivative. to do: why wouldn't it be 1:end-1 ?   
    % plot
    posteriorLine = plot(axPosterior,1./plausibleLambdas,posteriorPDF);
    posteriorLine.LineWidth = 2;
    posteriorLine.Color = [1 0 0];   
    posteriorLine.Visible = 'on';
end

function maxLikeLine = makeMaxLikeLine(axPosterior)
    maxLikeLine = plot(axPosterior,[axPosterior.XLim(1) axPosterior.XLim(1)],axPosterior.YLim);
    maxLikeLine.LineWidth = 3;
    maxLikeLine.Color = [0 1 0];
    maxLikeLine.Visible = 'off';

end

function makePosteriorAxisLegend(axPosterior,posteriorLine,maxLikeLine)
    textStrings = {'prior/posterior','max. like.'};
    textColors = [posteriorLine.Color; maxLikeLine.Color]; 
    compiledEntries = [];
    for entryI = 1:length(textStrings)
        currentString = textStrings{entryI};
        currentColor = textColors(entryI,:);        
        latexEntry = ['\color[rgb]{' num2str(currentColor(1))  ',' num2str(currentColor(2)) ',' num2str(currentColor(3)) '}' currentString];
        compiledEntries = [compiledEntries '    ' latexEntry];
    end    
    axPosterior.Title.String = [compiledEntries];
end

function setupEstimateTrackingAxis(axEstimateTracking,dataPoints,plausibleLambdas)
    axEstimateTracking.XLim = [-1 length(dataPoints)+1];
    axEstimateTracking.XLabel.String = 'n';
    axEstimateTracking.YLim = [min(plausibleLambdas) max(plausibleLambdas)];
    axEstimateTracking.YLabel.String = '\lambda'; axEstimateTracking.YLabel.FontSize = 18;
    axEstimateTracking.XLabel.String = '{\it n}'; axEstimateTracking.XLabel.FontSize = 18;
end

function setEstimateTrackingAxisTitle(axEstimateTracking,theta)
    compiledString = ['\theta = ['];
    for thetaI = 1:length(theta)
        compiledString = [compiledString ' ' num2str(theta(thetaI)) ' '];   
    end
    axEstimateTracking.Title.String = [compiledString ']'];
    axEstimateTracking.Title.HorizontalAlignment =  'center';
end

function estimateTrackingAxisLegend = makeEstimateTrackingAxisLegend(axEstimateTracking)
    fontSize = 14;
    textStrings = {'- - true','\fontsize{25}\cdot \fontsize{14}estimate'}; % still unsure how to get a good dot
    textColors = [[.5 .5 .5]; [.5 .5 .5]];
    ySpacing = diff(axEstimateTracking.YLim) * .125; % just a fraction the y-axis span (i.e., data-units) to keep it normalized.
    legendX = axEstimateTracking.XLim(2) - (.05*axEstimateTracking.XLim(2));
    legendY = axEstimateTracking.YLim(2) - (.5*ySpacing);
    for entryI = 1:length(textStrings)
       currentEntry = text(axEstimateTracking, legendX, legendY,textStrings{entryI}); 
       currentEntry.HorizontalAlignment = 'right';
       currentEntry.Units = 'normalized'; % in case user sets y-axis to update based on estimated pdf
       currentEntry.Color = textColors(entryI,:);
       currentEntry.FontName = 'Arial'; currentEntry.FontSize = fontSize;
       estimateTrackingAxisLegend{entryI} = currentEntry;
       legendY = legendY - ySpacing;
    end
end

function [trueLambdaLine] = makeTrueParameterLine(axEstimateTracking,lambda)
    trueLambdaLine = plot(axEstimateTracking, axEstimateTracking.XLim, [lambda lambda]);
    trueLambdaLine.Color = [0 0 1]; 
    trueLambdaLine.LineWidth = 1.5; 
    trueLambdaLine.LineStyle = '--';
end

function [lambdaPointArray] = makeEstimateScatterArrays(axEstimateTracking,dataPoints)
    timer = tic;
    disp('making point array')
    for dataI = 1:length(dataPoints)+1 % add one to account for initial values from prior        
        lambdaPoint = scatter(axEstimateTracking,dataI,axEstimateTracking.YLim(1));    
        lambdaPoint.MarkerFaceColor = [0 0 1]; lambdaPoint.MarkerEdgeColor = 'none'; 
        lambdaPoint.Visible = 'off';
        lambdaPointArray(dataI) = lambdaPoint;
    end 
    disp(['finished: ' num2str(toc(timer))])
end

