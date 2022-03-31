% BJD. For testing python code outputs against the original matlab script
% outputs for the normal conjugate case.
% The directory preceding this one should contain the matlab functions for
% the analysis (either directly or in one of it's subfolders). Run this script first,
% to generate a data file, and then run the python script
% (testNormal_part2.py) to test whether the class produces matching output.
% This code won't be terribly explanatory, but all details should be
% clarified in the primary scripts/functions. 

addpath(genpath(fileparts(pwd)),'-begin')
%% set number of datasets to generate/samples per, etc
dataStruct = struct;
numDatasets = 10;
numSamplesPerDataset = 50;
dataSigma = 2;

%% generate data
dataStruct.datasets = nan(numDatasets,numSamplesPerDataset);
for dataI = 1:numDatasets
    dataStruct.datasets(dataI,:) = normrnd(dataI,dataSigma,1,numSamplesPerDataset); % just using the index to set the mean to give variability
end

%% run analysis on datasets, storing output
prior = [0 0 -.5 0];
for dataI = 1:numDatasets
    currData = dataStruct.datasets(dataI,:);
    theta = prior;
    for datumI = 1:length(currData)
        datum = currData(datumI);
        theta = normalgamma_update(datum, theta);
    end
    %% get estimate
    plausibleMus = linspace(-4*dataSigma,4*dataSigma,20) + dataI; 
    plausibleTaus = linspace(.001,(1/(dataSigma^2))*4.5,20);
    posteriorPDF = normalgamma(plausibleMus,plausibleTaus,theta);
    posteriorSize = size(posteriorPDF);
    [M,I] = max(posteriorPDF(:)); 
    [r,c] = ind2sub(posteriorSize,I);
    bestMu = plausibleMus(r);
    bestTau = plausibleTaus(c);   
    %% store data
    dataStruct.bestMus(dataI,1) = bestMu;
    dataStruct.bestTaus(dataI,1) = bestTau;
    dataStruct.finalThetas(dataI,1:4) = theta;
end

%% save as a structure for loading in python (scipy io)
save('testNormal_DataStruct','dataStruct')




