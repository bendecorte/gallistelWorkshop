% Example animation for normal gamma posterior function.
% All code by BJD.
% The code-syntax is 'weird' because opengl rendering is meant for updating 
% existing graphic-objects, rather than repeatedly generating new ones. 
% The relationship is similar to that between preallocating a matrix and overall speed.
% For fast/smooth graphics, you create the plot object first (e.g., p = plot(1:n)), then 
% update its data using structure/object-syntax (e.g., p.YData = [1:n]*2). Repeated calls to 
% plot functions (e.g., plot(1:n); plot([1:n]*2)) will produce slow/inconsistent speed, as they typically generate new 
% plot objects at each call. So this code generates everything up-front, then updates during the animation loop. 

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
mu = 2;
sigma = .2;
dataPoints = normrnd(mu,sigma,1,numDataPoints);

%% prior parameters
%theta = [0 0 4 .37]; % INFORMATIVE prior; informs as to plausible value for tau
theta = [0 0 -.5 0]; % JEFFREYS prior

%% setup figure / make axes
figAnimation = makeAnimationFigure();
[axSource,axPosterior,axEstimateTracking] = makeAnimationAxes();

%% source axis (comes second because it needs estiamtes from the posterior setup above)
sourceXValues = linspace(-4*sigma,4*sigma,100) + mu; 
setupSourceAxis(axSource,sourceXValues)
[sourceLine,sourceEstimateLine,dataLine] = makeSourceAxisLines(axSource,sourceXValues,mu,sigma);
axSource.YLim = [0 max([sourceLine.YData sourceEstimateLine.YData])*1.5];
sourceAxisLegend = makeSourceAxisLegend(axSource,sourceLine,sourceEstimateLine,dataLine); % for flexibility, actually a cell array of text-objects, rather than typical axis legend

%% prior/posterior axis setup
numMeshPoints = 200;
plausibleMus = linspace(-4*sigma,4*sigma,numMeshPoints) + mu; % Mus/Tau values considered when computing the pdf
plausibleTaus = linspace(.001,(1/(sigma^2))*4.5,numMeshPoints); % could be a more robust way to set the limit here. sort of arbitrary.
plausibleSigmas = sqrt(1./plausibleTaus);
setupPosteriorAxis(axPosterior,plausibleTaus,plausibleMus)
[posteriorMesh,posteriorPDF,plausibleMuGrid,plausibleTauGrid] = makePosteriorMesh(axPosterior,theta,plausibleMus,plausibleTaus); % due to complexity, make each plot object for this axis with separate fuction

maxLikeLine = makeMaxLikeLine(axPosterior,plausibleMus,plausibleTaus,posteriorPDF);
[marginalMuLine,marginalTauLine] = makeMarginalLines(axPosterior,plausibleMus,plausibleTaus,posteriorPDF);
makePosteriorAxisLegend(axPosterior,posteriorMesh,maxLikeLine,marginalMuLine);

%% parameter history axis setup
setupEstimateTrackingAxis(axEstimateTracking,dataPoints,plausibleMus,plausibleSigmas);
[trueMuLine,trueSigmaLine] = makeTrueParameterLines(axEstimateTracking,mu,sigma);
[muPointArray,sigmaPointArray] = makeEstimateScatterArrays(axEstimateTracking,dataPoints); % basically, preallocating scatter plots for each iteration
estimateTrackingAxisLegend = makeEstimateTrackingAxisLegend(axEstimateTracking); % I take no shame in how long that function name is. 

for dataI = 0:length(dataPoints)
    updateTimer = tic;
    %% update parameter vector (for first iteration, just draw based on starting values)
    if dataI>0
        currentDataPoint = dataPoints(dataI);
        theta = normalgamma_update_animation(currentDataPoint, theta);
        dataLine.XData = [currentDataPoint currentDataPoint];
        if dataI==1; dataLine.Visible = 'on'; sourceEstimateLine.Visible = 'on'; end % show line when first real value is drawn
    end

    %% segargate the parameter vector (for clarity)
    estimatedMu = theta(1);
    lambda = theta(2); % aka nu (in e.g. NormGamPost)
    alpha  = theta(3);
    beta   = theta(4);
    
    %% update the posterior axis plot objects
    % get the posterior at all values of interest
    posteriorPDF = gampdf(plausibleTauGrid, alpha, 1/beta) .* normpdf(plausibleMuGrid, estimatedMu, 1./sqrt(lambda.*plausibleTauGrid));    
    % PDF mesh
    posteriorMesh.ZData = posteriorPDF;
    % maximum likelihood
    posteriorPDFSize = size(posteriorPDF);
    [maxPDFValues,maxPDFIndices] = max(posteriorPDF(:)); % peak probability
    [maxMuIndex,maxTauIndex] = ind2sub(posteriorPDFSize,maxPDFIndices);
    maxLikeLine.XData = [plausibleTaus(maxTauIndex) plausibleTaus(maxTauIndex)];
    maxLikeLine.YData = [plausibleMus(maxMuIndex) plausibleMus(maxMuIndex)];
    maxLikeLine.ZData = [0 max(maxPDFValues)*1.1];   
    % Mu's marginal distribution
    marginalMuLine.XData = ones(size(plausibleMus))*(axPosterior.XLim(2) - [.01*axPosterior.XLim(2)] ); % subtract a small value to make sure line is in view/on plot
    marginalMuLine.YData = plausibleMus;
    marginalMuZ = sum(posteriorPDF,2); % need to spell this one out before replacing data
    marginalMuZ = (marginalMuZ / max(marginalMuZ)) * max(maxPDFValues); % not quite sure if you should scale to the overall max or the max along the mu component. Same thing?
    marginalMuLine.ZData = marginalMuZ;    
    % Tau's marginal distribution
    marginalTauLine.XData = plausibleTaus;
    marginalTauLine.YData = ones(size(plausibleTaus))*(axPosterior.YLim(2) - [.01*axPosterior.YLim(2)] ); % see above
    marginalTauZ = sum(posteriorPDF,1); marginalTauZ = [marginalTauZ / max(marginalTauZ)] * max(max(posteriorPDF));
    marginalTauLine.ZData = marginalTauZ;
    
    %% draw the current estimate of the source distribution, given the parameters from the updated posterior
    currentMuEstimate = plausibleMus(maxMuIndex);
    currentTauEstimate = plausibleTaus(maxTauIndex);
    currentSigmaEstimate = sqrt(1/currentTauEstimate);
    currentSourceEstimatePDF = normpdf(sourceXValues,currentMuEstimate,currentSigmaEstimate);
    sourceEstimateLine.YData = currentSourceEstimatePDF;
    axSource.YLim = [0 max([sourceLine.YData sourceEstimateLine.YData])*1.2];
    
    %% plot estimate on bottom panel
    muPointArray(dataI+1).YData = currentMuEstimate;
    muPointArray(dataI+1).Visible = 'on'; % when preallocated, all points are set to be invisible
    sigmaPointArray(dataI+1).YData = currentSigmaEstimate; 
    sigmaPointArray(dataI+1).Visible = 'on';
    
    %% show current theta on the history axis (best place for this?)
    setEstimateTrackingAxisTitle(axEstimateTracking,round(theta,2))
    
    %% render and pause
    drawnow expose
    pause(updateSpeedSecs)
    
    %% check rendering time
    if showUpdateTime; disp(['update timer: ' num2str(toc(updateTimer))]); end
end

%% supporting functions\
function theta = normalgamma_update_animation(D, theta0)
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
        currentAx.FontSize = 12;
        currentAx.FontWeight = 'Bold';
        currentAx.XLabel.FontSize = 12; currentAx.YLabel.FontSize = 12;
    end
end

function setupSourceAxis(axSource,sourceXValues)
    axSource.XLim = [min(sourceXValues) max(sourceXValues)];
    axSource.XLabel.String = 'latency (secs)'; axSource.YLabel.String = 'density';
    axSource.Title.String = 'Source (true vs. estimate)';
end

function [sourceLine,sourceEstimateLine,dataLine] = makeSourceAxisLines(axSource,sourceXValues,mu,sigma)
    % true source 
    sourcePDF = normpdf(sourceXValues,mu,sigma);
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
    dataLine = plot(axSource,[mu mu],axSource.YLim);
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
    legendX = axSource.XLim(1) + .125;
    legendY = axSource.YLim(2) - (.5*ySpacing);
    for entryI = 1:length(textStrings)
       currentEntry = text(axSource, legendX, legendY,textStrings{entryI}); 
       currentEntry.Units = 'normalized'; % in case user sets y-axis to update based on estimated pdf
       currentEntry.Color = textColors(entryI,:);
       currentEntry.FontName = 'Arial'; currentEntry.FontSize = fontSize;
       sourceAxisLegend{entryI} = currentEntry;
       legendY = legendY - ySpacing;
    end
end

function setupPosteriorAxis(axPosterior,plausibleTaus,plausibleMus)
    axPosterior.View = [-49.0212 26.9866]; % chosen manually
    rotate3d(axPosterior) % pre-set interaction mode to 'rotate' for inspection during the animation
    axPosterior.XLim = [min(plausibleTaus) max(plausibleTaus)];
    axPosterior.YLim = [min(plausibleMus)  max(plausibleMus)];
    axPosterior.XLabel.String = '\tau  (1/\sigma^2)'; 
    axPosterior.XLabel.FontSize  = 18;
    axPosterior.YLabel.String = '\mu';
    axPosterior.YLabel.FontSize = axPosterior.XLabel.FontSize;
    axPosterior.ZLabel.String = 'density';    
end

function [posteriorMesh,posteriorPDF,plausibleMuGrid,plausibleTauGrid] = makePosteriorMesh(axPosterior,theta,plausibleMus,plausibleTaus)
    % unpack current theta for clarity
    estimatedMu = theta(1);
    lambda = theta(2); % aka nu (in e.g. NormGamPost)
    alpha  = theta(3);
    beta   = theta(4);
    % make the mesh
    [plausibleMuGrid,plausibleTauGrid] = ndgrid(plausibleMus,plausibleTaus); 
    posteriorPDF = gampdf(plausibleTauGrid, alpha, 1/beta) .* normpdf(plausibleMuGrid, estimatedMu, 1./sqrt(lambda.*plausibleTauGrid));
    posteriorMesh = mesh(axPosterior,plausibleTaus,plausibleMus,posteriorPDF);
    posteriorMesh.FaceAlpha = .1;
end

function [maxLikeLine] = makeMaxLikeLine(axPosterior,plausibleMus,plausibleTaus,posteriorPDF)
    posteriorPDFSize = size(posteriorPDF);
    [~,maxIndices] = max(posteriorPDF(:)); % peak probability
    [maxMuIndex,maxTauIndex] = ind2sub(posteriorPDFSize,maxIndices);
    maxLikeLine = plot3(axPosterior,[plausibleTaus(maxTauIndex) plausibleTaus(maxTauIndex)],[plausibleMus(maxMuIndex) plausibleMus(maxMuIndex)],[0 max(max(posteriorPDF))*1.1]);
    maxLikeLine.LineWidth = 3;
    maxLikeLine.Color = [0 1 0];
end

function [marginalMuLine,marginalTauLine] = makeMarginalLines(axPosterior,plausibleMus,plausibleTaus,posteriorPDF)
    % mu
    marginalMuX = ones(size(plausibleMus))*(axPosterior.XLim(2) - [.01*axPosterior.XLim(2)]);
    marginalMuY = plausibleMus;
    marginalMuZ = sum(posteriorPDF,2); marginalMuZ = [marginalMuZ / max(marginalMuZ)] * max(max(posteriorPDF));
    marginalMuLine = plot3(axPosterior,marginalMuX,marginalMuY,marginalMuZ);
    marginalMuLine.Color = [.5 .5 .5]; 
    marginalMuLine.LineWidth = 4;
    % tau
    marginalTauX = plausibleTaus;
    marginalTauY = ones(size(plausibleTaus))*(axPosterior.YLim(2) - [.01*axPosterior.YLim(2)]);
    marginalTauZ = sum(posteriorPDF,1); marginalTauZ = [marginalTauZ / max(marginalTauZ)] * max(max(posteriorPDF));
    marginalTauLine = plot3(axPosterior,marginalTauX,marginalTauY,marginalTauZ);
    marginalTauLine.Color = marginalMuLine.Color; 
    marginalTauLine.LineWidth = 4;
end

function makePosteriorAxisLegend(axPosterior,posteriorMesh,maxLikeLine,marginalMuLine)
    textStrings = {'prior/posterior','max. like.','marginal'};
    textColors = [[0 0 1]; maxLikeLine.Color; marginalMuLine.Color]; % hard-code the posterior color, as it's ambiguous. 
    compiledEntries = [];
    for entryI = 1:length(textStrings)
        currentString = textStrings{entryI};
        currentColor = textColors(entryI,:);        
        latexEntry = ['\color[rgb]{' num2str(currentColor(1))  ',' num2str(currentColor(2)) ',' num2str(currentColor(3)) '}' currentString];
        compiledEntries = [compiledEntries '    ' latexEntry];
    end    
    axPosterior.Title.String = [compiledEntries];
end

function setupEstimateTrackingAxis(axEstimateTracking,dataPoints,plausibleMus,plausibleSigmas)
    axEstimateTracking.XLim = [-1 length(dataPoints)+1];
    axEstimateTracking.XLabel.String = 'n';

    yyaxis left
    tempAx = gca; % yy plots are a bit weird to code explicitly with. 
    tempAx.YLim = [min(plausibleMus) max(plausibleMus)];
    tempAx.YLabel.String = '\mu'; tempAx.YLabel.FontSize = 18;
    tempAx.XLabel.String = '{\it n}'; tempAx.XLabel.FontSize = 18;

    yyaxis right
    tempAx = gca;
    tempAx.YLabel.String = '\sigma'; tempAx.YLabel.FontSize = 18;
    tempAx.YLim = [min(plausibleSigmas) std(dataPoints)*2.5]; 
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

function [trueMuLine,trueSigmaLine] = makeTrueParameterLines(axEstimateTracking,mu,sigma)
    axes(axEstimateTracking)
    yyaxis left
    tempAx = gca;
    trueMuLine = plot(tempAx.XLim,[mu mu]);
    trueMuLine.Color = [0 0 1]; 
    trueMuLine.LineWidth = 1.5; 
    trueMuLine.LineStyle = '--';
    
    yyaxis right
    tempAx = gca;
    trueSigmaLine = plot(tempAx.XLim,[sigma sigma]);
    trueSigmaLine.Color = [1 0 0]; 
    trueSigmaLine.LineWidth = trueMuLine.LineWidth; 
    trueSigmaLine.LineStyle = trueMuLine.LineStyle;
end

function [muPointArray,sigmaPointArray] = makeEstimateScatterArrays(axEstimateTracking,dataPoints)
    timer = tic;
    disp('making point arrays')
    axes(axEstimateTracking)
    yyaxis left % note: doing both axes in one loop will be slower. yyaxis left/right calls can be sluggish
    tempAx = gca;
    for dataI = 1:length(dataPoints)+1 % add one to account for initial values from prior        
        muPoint = scatter(tempAx,dataI,tempAx.YLim(1));    
        muPoint.MarkerFaceColor = [0 0 1]; muPoint.MarkerEdgeColor = 'none';
        muPoint.Visible = 'off';
        muPointArray(dataI) = muPoint;
    end
    yyaxis right
    tempAx = gca;
    for dataI = 1:length(dataPoints)+1
        sigmaPoint = scatter(tempAx,dataI,tempAx.YLim(1));
        sigmaPoint.MarkerFaceColor = [1 0 0]; sigmaPoint.MarkerEdgeColor = 'none';
        sigmaPoint.Visible = 'off';
        sigmaPointArray(dataI) = sigmaPoint;
    end
    disp(['finished: ' num2str(toc(timer))])
end

