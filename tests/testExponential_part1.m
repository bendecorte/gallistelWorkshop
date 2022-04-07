% BJD. For testing python code outputs against the original matlab script
% outputs for the exp conjugate case.
% The directory preceding this one should contain the matlab functions for
% the analysis (either directly or in one of it's subfolders). Run this script first,
% to generate a data file, and then run the python script
% (testExponential_part2.py) to test whether the class produces matching output.
% This code won't be terribly explanatory, but all details should be
% clarified in the primary scripts/functions. 

addpath(genpath(fileparts(pwd)),'-begin')
%% set number of datasets to generate/samples per, etc
dataStruct = struct;
numDatasets = 10;
numSamplesPerDataset = 50;

%% generate data
dataStruct.datasets = nan(numDatasets,numSamplesPerDataset);
for dataI = 1:numDatasets
    dataStruct.datasets(dataI,:) = cumsum(exprnd(dataI/10,1,numSamplesPerDataset)); % just using the index to set the mean to give variability
end

%% run analysis on datasets, storing output
prior = [.5 0];
for dataI = 1:numDatasets
    currData = dataStruct.datasets(dataI,:);
    theta = prior;
    for datumI = 1:length(currData)
        datum = currData(datumI);
        [mxLkLam,mnPostLam,theta,postgam,postmu] = expoupdate(datumI,datum,theta,500);
    end
    %% store data
    dataStruct.maxLams(dataI,1) = mxLkLam;
    dataStruct.muLams(dataI,1) = mnPostLam;
    %dataStruct.posteriors(dataI,:) = postgam(:,2); % they only differ in terms of x axis values
    dataStruct.finalThetas(dataI,:) = theta;
end
%% save as a structure for loading in python (scipy io)
save('testExponential_DataStruct','dataStruct')




