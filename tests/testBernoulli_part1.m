% BJD. For testing python code outputs against the original matlab script
% outputs for the exp conjugate case.
% The directory preceding this one should contain the matlab functions for
% the analysis (either directly or in one of it's subfolders). Run this script first,
% to generate a data file, and then run the python script
% (testExponential_part2.py) to test whether the class produces matching output.
% This code won't be terribly explanatory, but all details should be
% clarified in the primary scripts/functions. 
clear all
addpath(genpath(fileparts(pwd)),'-begin')
%% set number of datasets to generate/samples per, etc
dataStruct = struct;
numDatasets = 10;
numSamplesPerDataset = 50;

%% generate data
dataStruct.datasets = nan(numDatasets,numSamplesPerDataset);
for dataI = 1:numDatasets
    p = rand;
    dataStruct.datasets(dataI,:) = binornd(1,p,1,numSamplesPerDataset); % just using the index to set the mean to give variability
end

%% run analysis on datasets, storing output
% single distribution case
prior = [.5 .5];
for dataI = 1:numDatasets
    currData = dataStruct.datasets(dataI,:);
    theta = prior;
    for datumI = 1:length(currData)
        datum = currData(datumI);
        [Pest1,theta] = BayesEstBernP(datum,theta);
    end
    %% store data
    dataStruct.ps(dataI,1) = Pest1;
    dataStruct.finalThetas(dataI,:) = theta;
end

% null value
nullValue = .2;
theta = [.5 .5];
dataStruct.nullJoints = [];
for dataI = 1:numDatasets
    currData = dataStruct.datasets(dataI,:);
    [Pest1,theta1,Pdist1,Pest2,theta2,Pdist2,Pjnt] = BayesEstBernP(currData,theta,nullValue);
    dataStruct.nullJoints = [dataStruct.nullJoints; ones(size(Pjnt,1),1)*dataI Pjnt];
end

% two distributions/datasets
theta = [.5 .5];
dataStruct.dualJoints = [];
for dataI = 1:numDatasets-1
    currData1 = dataStruct.datasets(dataI,:);
    currData2 = dataStruct.datasets(dataI+1,:);
    [Pest1,theta1,Pdist1,Pest2,theta2,Pdist2,Pjnt] = BayesEstBernP(currData1,theta,currData2);
    dataStruct.dualJoints = [dataStruct.dualJoints; ones(size(Pjnt,1),1)*dataI Pjnt];
end

%% save as a structure for loading in python (scipy io)
save('testBernoulli_DataStruct','dataStruct')




