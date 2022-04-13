% Example animation for Bernoulli distribution
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
updateSpeedSecs = .25;

%% source parameters
p = .2; 
dataPoints = binornd(1,p,1,numDataPoints);

%% prior parameters
theta = [.5 .5]; % JEFFREYS prior

%% setup figure / make axes
figAnimation = makeAnimationFigure();
[axSource,axPosterior,axEstimateTracking] = makeAnimationAxes();

%% source axis (comes second because it needs estiamtes from the posterior setup above)
sourceXValues = [0 1];
setupSourceAxis(axSource,sourceXValues)
[trueFailBar,trueSuccessBar,estimateFailBar,estimateSuccessBar] = makeSourceAxisBars(axSource,sourceXValues,p,theta); % no formal 'data' plot object for this one. tricky representing that on a bar graph
yVerticesToChange = estimateFailBar.Vertices(:,2)~=0; % just the vertices corresponding to the top of the estimate bars (the 0's just correspond to the bottom of the bar). 
sourceAxisLegend = makeSourceAxisLegend(axSource,trueFailBar,estimateFailBar); % for flexibility, actually a cell array of text-objects, rather than typical axis legend

%% prior/posterior axis setup
setupPosteriorAxis(axPosterior)
[posteriorLine] = makePosteriorLine(axPosterior,theta); % note that lamdas get slight correction due to the way the pdf is computed (diff of cdf). 
maxLikeLine = makeMaxLikeLine(axPosterior,posteriorLine);
makePosteriorAxisLegend(axPosterior,posteriorLine,maxLikeLine);

%% parameter history axis setup
setupEstimateTrackingAxis(axEstimateTracking,dataPoints);
[trueLambdaLine] = makeTrueParameterLine(axEstimateTracking,p);
[pPointArray] = makeEstimateScatterArrays(axEstimateTracking,dataPoints); % basically, preallocating scatter plots for each iteration
estimateTrackingAxisLegend = makeEstimateTrackingAxisLegend(axEstimateTracking); % I take no shame in how long that function name is. 
for dataI = 1:length(dataPoints)
    updateTimer = tic;
    %% update parameter vector (for first iteration, just draw based on starting values)
    if dataI>0
        currentDataPoint = dataPoints(dataI);
        indicateCurrentDatum(axSource,currentDataPoint,.1)
        [pEstimate,theta,posteriorPDF] = BayesEstBernP_animation(currentDataPoint,theta);
        dataLine.XData = [currentDataPoint currentDataPoint];
        if dataI==1; dataLine.Visible = 'on'; sourceEstimateLine.Visible = 'on'; maxLikeLine.Visible = 'on'; end % show line when first real value is drawn
    end

    %% update the posterior axis plot objects
    % get the posterior at all values of interest
    posteriorPDF = posteriorPDF(:,2); 
    % PDF 
    posteriorLine.YData = posteriorPDF;
    % maximum likelihood
    maxLikeLine.XData = [pEstimate pEstimate];
    maxLikeLine.YData = axPosterior.YLim;
    axPosterior.YLim = [0 max(posteriorLine.YData)*1.2];

    %% draw the current estimate of the source distribution, given the parameters from the updated posterior
    estimateSuccessBar.Vertices(yVerticesToChange,2) = pEstimate;
    estimateFailBar.Vertices(yVerticesToChange,2) = 1-pEstimate;
    
    %% plot estimate on bottom panel
    pPointArray(dataI+1).YData = pEstimate;
    pPointArray(dataI+1).Visible = 'on'; % when preallocated, all points are set to be invisible
    
    %% show current theta on the history axis (best place for this?)
    setEstimateTrackingAxisTitle(axEstimateTracking,round(theta,2))
    
    %% render and pause
    drawnow expose
    pause(updateSpeedSecs)
    
    %% check rendering time
    if showUpdateTime; disp(['update timer: ' num2str(toc(updateTimer))]); end
end






%% supporting functions
function [Pest1,theta1,Pdist1] = BayesEstBernP_animation(D1,theta0)
    %% minimal function for single-distribution case. 
    D1 = logical(D1);
    theta1 = [sum(D1) sum(~D1)] + theta0; % updating
    Pdist1(:,1) = (.005:.01:.995)'; % support for beta posterior(s)
    Pdist1(:,2) = betapdf(Pdist1(:,1),theta1(1),theta1(2)); % probability
    % densities for the first posterior
    Pdist1(:,2) = Pdist1(:,2)/sum(Pdist1(:,2)); % normalizing
    Pest1 = sum(Pdist1(:,2).*Pdist1(:,1)); % probability-weighted sum, i.e.,
    % the mean of the posterior distribution

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
    axSource.XLim = [min(sourceXValues)-1 max(sourceXValues)+1];
    axSource.XTick = sourceXValues;
    axSource.XTickLabel = {'failure','success'};
    axSource.XLabel.String = 'outcomes'; 
    axSource.YLim = [0 1.1];
    axSource.YTick = [0 .5 1];
    axSource.YLabel.String = 'probability';
    axSource.Title.String = 'Source (true vs. estimate)';
end

function [trueFailBar,trueSuccessBar,estimateFailBar,estimateSuccessBar] = makeSourceAxisBars(axSource,sourceXValues,p,theta)
    barWidth = .3; % just hard coded for now
    % true source bars
    xCenter = sourceXValues(1) - (barWidth/2);
    trueFailBar = barFill(axSource,xCenter,1-p,barWidth);
    xCenter = sourceXValues(2) - (barWidth/2);
    trueSuccessBar = barFill(axSource,xCenter,p,barWidth);
    
    % get posterior estimate from initial theta (if possible under current prior parameters)
    % unpack theta
    alpha = theta(1);
    beta = theta(2);
    try
        % quick attempt at getting initial pEstimate from user's prior theta
        posteriorX = .005:.01:.995;
        posteriorPDF = betapdf(posteriorX,alpha,beta); % just a 
        posteriorPDF = posteriorPDF/sum(posteriorPDF);
        pEstimate = sum(posteriorX.*posteriorPDF);
    catch
        % set it arbitrarily if user did something weird (e.g., passed value that's uncomputable under the beta dist.) 
        disp('Unable to compute a posterior from initial theta. Setting inital estimates to .5 in figure.')
        pEstimate = .5;
    end
    % estimate source bars
    xCenter = sourceXValues(1) + (barWidth/2);
    estimateFailBar = barFill(axSource,xCenter,1-pEstimate,barWidth);
    estimateFailBar.FaceColor = ones(1,3)*.6;
    xCenter = sourceXValues(2) + (barWidth/2);
    estimateSuccessBar = barFill(axSource,xCenter,pEstimate,barWidth);
    estimateSuccessBar.FaceColor = estimateFailBar.FaceColor;

end

function barObj = barFill(axTarget,x,y,barWidth)
    % the standard bar function would be hard to work with in this example, so this
    % function generates individual bars as fill objects, which are easier
    % to manipulate individually
    xSpacing = barWidth/2;
    rect = [x-xSpacing 0; x-xSpacing y; x+xSpacing y; x+xSpacing 0]; % start at bottom left and move clockwise
    barObj = fill(axTarget,rect(:,1),rect(:,2),[0 0 0]); % x points, y points, then color (black as default)
    barObj.EdgeColor = 'none';
end

function sourceAxisLegend = makeSourceAxisLegend(axSource,sourceBar,estimateBar)
    fontSize = 14;
    textStrings = {'true','estimate'};
    textColors = [sourceBar.FaceColor; estimateBar.FaceColor];
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

function indicateCurrentDatum(axSource,currentDatum,blinkPauseSecs)
    failString = axSource.XTickLabels{1}; failString = failString(failString~='*');
    successString = axSource.XTickLabels{2}; successString = successString(successString~='*');
    axSource.XTickLabels = {failString;successString};
    pause(blinkPauseSecs)
    if currentDatum==0
        failString = ['*' failString '*'];
    end
    if currentDatum==1
        successString = ['*' successString '*'];
    end  
    axSource.XTickLabels = {failString;successString};
end

function setupPosteriorAxis(axPosterior)
    axPosterior.XLim = [0 1];
    axPosterior.XLabel.String = '\itp'; 
    axPosterior.YLabel.String = 'density';
    axPosterior.YLabel.FontSize = axPosterior.XLabel.FontSize;   
end

function [posteriorLine] = makePosteriorLine(axPosterior,theta)
    % unpack theta for clarity
    alpha = theta(1);
    beta = theta(2);
    % compute pdf
    plausibleProbabilities = .005:.01:.995; % to do: 'plausible' isn't really the right word. 
    posteriorPDF = betapdf(plausibleProbabilities,alpha,beta);
    posteriorPDF = posteriorPDF / sum(posteriorPDF);
    % plot
    posteriorLine = plot(axPosterior,plausibleProbabilities,posteriorPDF);
    posteriorLine.LineWidth = 2;
    posteriorLine.Color = [1 0 0];   
    posteriorLine.Visible = 'on';
end

function maxLikeLine = makeMaxLikeLine(axPosterior,posteriorLine)
    xValues = posteriorLine.XData;
    yValues = posteriorLine.YData;
    estimatedP = sum(xValues.*yValues);
    maxLikeLine = plot(axPosterior,[estimatedP estimatedP],axPosterior.YLim);
    maxLikeLine.LineWidth = 3;
    maxLikeLine.Color = [0 1 0];
    maxLikeLine.Visible = 'on';

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

function setupEstimateTrackingAxis(axEstimateTracking,dataPoints)
    axEstimateTracking.XLim = [-1 length(dataPoints)+1];
    axEstimateTracking.XLabel.String = 'n';
    axEstimateTracking.YLim = [0 1];
    axEstimateTracking.YLabel.String = '\itp'; axEstimateTracking.YLabel.FontSize = 18;
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

function [trueLambdaLine] = makeTrueParameterLine(axEstimateTracking,p)
    trueLambdaLine = plot(axEstimateTracking, axEstimateTracking.XLim, [p p]);
    trueLambdaLine.Color = [0 0 1]; 
    trueLambdaLine.LineWidth = 1.5; 
    trueLambdaLine.LineStyle = '--';
end

function [pPointArray] = makeEstimateScatterArrays(axEstimateTracking,dataPoints)
    timer = tic;
    disp('making point array')
    for dataI = 1:length(dataPoints)+1 % add one to account for initial values from prior        
        lambdaPoint = scatter(axEstimateTracking,dataI,axEstimateTracking.YLim(1));    
        lambdaPoint.MarkerFaceColor = [0 0 1]; lambdaPoint.MarkerEdgeColor = 'none'; 
        lambdaPoint.Visible = 'off';
        pPointArray(dataI) = lambdaPoint;
    end 
    disp(['finished: ' num2str(toc(timer))])
end

