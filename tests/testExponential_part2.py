# BJD. Code for testing whether python class for the normal conjugate case produces matching outputs to the original matlab functions.
# Run testNormal_part1 first, which will save a data structure storing the matlab output. Then, run this one, pasting the Normal class at the top with the modified one you created. 
# Hopefully the script will be self-explanatory, but email BJD for clarifications.

# MIT License
# Copyright (c) 2022 bendecorte
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os, copy
import numpy as np
import scipy.io 
import scipy.stats
import matplotlib.pyplot as plt

class Exponential():
    def __init__(self):
        self.jeffreysPrior = np.array([.5,0])    
    def updateTheta(self,n,data,thetaCurrent):
        self.checkUpdateInputs(data,thetaCurrent)
        alphaCurrent,betaCurrent = self.unpackTheta(thetaCurrent)
        alphaNew = alphaCurrent + n
        betaNew = betaCurrent + data
        thetaNew = np.array([alphaNew,betaNew])
        return thetaNew
    def checkUpdateInputs(self,data,thetaCurrent):
        if not isinstance(data, np.floating): # the ideal input
            if not isinstance(data, np.ndarray) or data.ndim!=1 and data.size>1: # basic check on array inputs
                raise Exception('data must be a numpy float or 1-d numpy array')
        if not isinstance(thetaCurrent, np.ndarray) or thetaCurrent.ndim!=1:
            raise Exception('theta must be 1-d numpy array')
    def unpackTheta(self,thetaCurrent):
        alpha = thetaCurrent[0]
        beta = thetaCurrent[1]
        return(alpha,beta)
    def getPosteriorStats(self,n,data,theta,numBins):
        alpha,beta = self.unpackTheta(theta)   
        minPosteriorX = scipy.stats.gamma.ppf(.001,alpha,scale = 1/beta) 
        maxPosteriorX = scipy.stats.gamma.ppf(.999,alpha,scale = 1/beta) 
        posteriorX = np.linspace(minPosteriorX,maxPosteriorX,numBins) 
        deltaX = posteriorX[1] - posteriorX[0] # granularity
        posteriorCDF = scipy.stats.gamma.cdf(posteriorX,alpha,scale = 1/beta) # to do: why not just call gamma directly here (rather than take derivative/normalize)?
        posteriorPDF = np.diff(posteriorCDF)
        posteriorPDF = posteriorPDF / np.sum(posteriorPDF)
        posteriorX = posteriorX[1:] - (deltaX / 2)
        # can plot with two different approaches for what goes on the x-axis
        lambdaX = posteriorX # everything computed with respect to lambda above. plotting this against the posterior PDF gives the gamma dist. over the exponential's rate-parameter
        muX = 1/posteriorX # but we can also express this with respect to 1/lambda, corresponding to different plausible means of the exponential
        bestLambda = np.sum(posteriorPDF*lambdaX)
        maxLikeLambda = n/data # doesn't take much work with exponential data
        return (bestLambda,maxLikeLambda,posteriorPDF,lambdaX,muX)
    def getKLDivergence(self,trueDistribution,approximatingDistribution):
        divergence = np.log(trueDistribution) - np.log(approximatingDistribution) + (approximatingDistribution/trueDistribution) - 1
        return divergence
    def makeFigure(self,n,dataValue,theta,numPosteriorBins):
        # get data
        alpha,beta = self.unpackTheta(theta) 
        bestLambda,maxLikeLambda,posteriorPDF,lambdaX,muX = exponentialConjugate.getPosteriorStats(n,dataValue,theta,numPosteriorBins)
        bestMu = 1/bestLambda
        sourceX = np.linspace(0,scipy.stats.expon.ppf(.99,scale = bestMu),200)
        estimatePDF = scipy.stats.expon.pdf(sourceX,scale = bestMu)
        # make figure
        fig = plt.figure(figsize=plt.figaspect(.5))  
        axSourceEstimate = fig.add_subplot(1,2,1)
        axPosterior = fig.add_subplot(1,2,2)

        # plot estimate pdf
        estimateLine, = axSourceEstimate.plot(sourceX,estimatePDF)
        estimateLine.set_linewidth(3)
        estimateLine.set_color([0,1,0])      
        axSourceEstimate.set_title('estimate')
        axSourceEstimate.set_xlim(np.min(sourceX),np.max(sourceX))
        axSourceEstimate.set_xlabel('data units')
        axSourceEstimate.set_ylabel('density')

        # plot posterior
        posteriorLine, = axPosterior.plot(muX,posteriorPDF)
        posteriorLine.set_linewidth(3)
        posteriorLine.set_color('red')
        muLine, = axPosterior.plot([bestMu,bestMu],axPosterior.get_ylim())
        muLine.set_linewidth(3)
        muLine.set_color([0,1,0])
        axPosterior.set_xlim(np.min(muX),np.max(muX))
        axPosterior.set_xlabel('mean inter-event interval')
        titleString = 'posterior: ' + r'$\theta$' + ' = ' + '[' + str(round(alpha,1)) + ', ' + str(round(beta,1)) + ']'
        axPosterior.set_title(titleString)

        # a few asthetics applied to both axes
        axisList = [axSourceEstimate,axPosterior]
        for axCurr in axisList:
            axCurr.spines['top'].set_visible(False)
            axCurr.spines['right'].set_visible(False)
            axCurr.set_ylim(0,axCurr.get_ylim()[1])
            yTicks = np.round(axCurr.get_yticks(),3)
            axCurr.set_yticks([0., np.max(yTicks)])
        muLine.set_ydata(axPosterior.get_ylim())

        # adjust positions
        axHeight = .855
        axWidth = .42
        colSpace = .075
        pos = [.07,.095,axWidth,axHeight]
        axSourceEstimate.set_position(pos,which = 'both')
        pos = copy.deepcopy(pos) # unsure if the copy will ever matter but just in case
        pos[0] = pos[0] + axWidth + colSpace
        axPosterior.set_position(pos)
        return (fig,axSourceEstimate,axPosterior,estimateLine)
    def addTrueSourcePlots(self,axSourceEstimate,estimateLine,mu):
        # plot source
        xlim = axSourceEstimate.get_xlim()
        sourceX = np.linspace(xlim[0],xlim[1],200)
        sourcePDF = scipy.stats.expon.pdf(sourceX,scale = mu)
        sourceLine, = axSourceEstimate.plot(sourceX,sourcePDF,'--')
        sourceLine.set_linewidth(3)        
        sourceLine.set_color('black')
        # correct y axis
        maxY = axSourceEstimate.get_ylim()[1]
        if maxY<np.max(sourcePDF): # should be rare, but needs correction
            newMaxY = np.max(sourcePDF)
            newMaxY = np.ceil(newMaxY*100) / 100 # for rounding
            axSourceEstimate.set_ylim(axSourceEstimate.get_ylim()[0],newMaxY)
            yTicks = np.array([0., newMaxY]) 
            axSourceEstimate.set_yticks(yTicks)
        # update legend/text
        axSourceEstimate.set_title(' ') # want to keep the space occupied by the legend, so don't delete
        textX = axSourceEstimate.get_xlim()[0] + (.1*np.diff(axSourceEstimate.get_xlim()))
        textY = axSourceEstimate.get_ylim()[1] * 1.0125
        estimateText = axSourceEstimate.text(textX,textY,'estimate')
        estimateText.set_color(estimateLine.get_color())
        textX = copy.deepcopy(textX) + (.25*np.diff(axSourceEstimate.get_xlim()))
        textY = copy.deepcopy(textY)
        trueText = axSourceEstimate.text(textX,textY,'true')
        trueText.set_color(sourceLine.get_color())

fileName = 'testExponential_DataStruct.mat'
floatPrecisionTolerance = 1e-10 # see loop below. allow for trivial deviations between matlab/python output related to factors such as double-precison, minor differences in distribution implementations, etc.

path = os.path.dirname(os.path.realpath(__file__))
fullFileName = os.path.join(path,fileName)

dataStruct = scipy.io.loadmat(fullFileName)
dataStruct = dataStruct['dataStruct'] # should be a structured np array with fields corresponding to those created in matlab (datasets,bestMus,bestTaus,and finalThetas)

datasets = dataStruct[0][0]['datasets'] # rows = datsets / cols = observations
matlabMaxLams = dataStruct[0][0]['maxLams']
matlabMuLams = dataStruct[0][0]['muLams']
#matlabPosteriors = dataStruct[0][0]['posteriors']
matlabThetas = dataStruct[0][0]['finalThetas']


exponentialConjugate = Exponential()
numEvaluationPoints = 500 # precision level for 
for dataI in range(np.shape(datasets)[0]):
    currDataset = datasets[dataI,:]
    pyTheta = exponentialConjugate.jeffreysPrior
    for datumI,datum in enumerate(currDataset):
        pyTheta = exponentialConjugate.updateTheta(datumI+1,datum,pyTheta)

    pyMuLam,pyMaxLam,posterior,_,_ = exponentialConjugate.getPosteriorStats(np.shape(currDataset)[0],currDataset[len(currDataset)-1],pyTheta,numEvaluationPoints)

    matlabTheta = matlabThetas[dataI,:]
    matlabMaxLam = matlabMaxLams[dataI,0]
    matlabMuLam = matlabMuLams[dataI,0]


    print(' ')
    print('dataset ' + str(dataI+1))
    print('difference theta',np.sum(matlabTheta - pyTheta))
    print('difference max Lam',matlabMaxLam - pyMaxLam)
    print('difference mu Lam',matlabMuLam - pyMuLam)

    if np.sum(matlabTheta - pyTheta) != 0: # should be exact match
        print('pyTheta',pyTheta)
        print('matlabTheta',matlabTheta)
        raise Exception('unacceptable deviation between matlab and python detected for theta. See output above.')
    if matlabMuLam - pyMuLam > floatPrecisionTolerance:
        print('pyMu',pyMuLam)
        print('matlabMu',matlabMuLam)
        raise Exception('unacceptable deviation between matlab and python detected for mu. See output above.')
    if matlabMaxLam - pyMaxLam > floatPrecisionTolerance:
        print('pyMaxLam',pyMaxLam)
        print('matlabMaxLam',matlabMaxLam)
        raise Exception('unacceptable deviation between matlab and python detected for tau. See output above.')

    









