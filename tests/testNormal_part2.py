# BJD. Code for testing whether python class for the normal conjugate case produces matching outputs to the original matlab functions.
# Run testNormal_part1 first, which will save a data structure storing the matlab output. Then, run this one, pasting the Normal class at the top with the modified one you created. 
# Hopefully the script will be self-explanatory, but email BJD for clarifications.

import os, copy
import numpy as np
import scipy.io 
import scipy.stats
import matplotlib.pyplot as plt

class Normal():
    def __init__(self):
        self.jeffreysPrior = np.array([0, 0, -.5, 0])    
        # note: if faster runtime needed. Could store other params inside object at startup (e.g., plausibleMus, grids, etc). The functions here will regenerate things like the mu/tau grids on the fly for flexibility/readability, which wouldn't be required. 
    def updateTheta(self,data,thetaCurrent):
        self.checkUpdateInputs(data,thetaCurrent)
        muCurrent,lambdaCurrent,alphaCurrent,betaCurrent = self.unpackTheta(thetaCurrent)
        n = data.size 
        muData = np.mean(data)
        varData = np.var(data)
        lambdaNew = lambdaCurrent + n 
        alphaNew = alphaCurrent + (n/2)
        muNew = ( (lambdaCurrent*muCurrent) + (n*muData) ) / (lambdaCurrent + n)
        betaNew = betaCurrent + 0.5*( ( (n*varData) + (lambdaCurrent*n*pow(muData-muNew,2)) ) / (lambdaCurrent+n) )
        thetaNew = np.array([muNew,lambdaNew,alphaNew,betaNew])
        return thetaNew
    def checkUpdateInputs(self,data,thetaCurrent):
        if not isinstance(data, np.floating): # the ideal input
            if not isinstance(data, np.ndarray) or data.ndim!=1 and data.size>1: # basic check on array inputs
                raise Exception('data must be a numpy float or 1-d numpy array')
        if not isinstance(thetaCurrent, np.ndarray) or thetaCurrent.ndim!=1:
            raise Exception('theta must be 1-d numpy array')
    def unpackTheta(self,thetaCurrent):
        mu = thetaCurrent[0]
        _lambda = thetaCurrent[1] # reserved in python
        alpha = thetaCurrent[2]
        beta = thetaCurrent[3]
        return(mu,_lambda,alpha,beta)
    def getPosteriorStats(self,plausibleMus,plausibleTaus,theta):
        mu,_lambda,alpha,beta = self.unpackTheta(theta)   
        plausibleMuGrid,plausibleTauGrid = np.meshgrid(plausibleMus,plausibleTaus,indexing = 'ij')   
        if alpha==0 and beta==0: # avoids a runtime error that'll happen when using a Jeffreys prior. Gamma pdf will be all nans in scipy (all inf in matlab). Just returning nan. 
            bestMu = np.nan
            bestTau = np.nan
            posteriorPDF = np.empty(shape = np.shape(plausibleMuGrid))
            posteriorPDF[:] = np.nan
        else:
            normPDF = scipy.stats.norm.pdf(plausibleMuGrid, mu, 1/np.sqrt(_lambda*plausibleTauGrid))
            gammaPDF = scipy.stats.gamma.pdf(plausibleTauGrid, alpha, scale = 1/beta)
            posteriorPDF = gammaPDF * normPDF
            bestMu,bestTau = self.getPosteriorMaxLike(posteriorPDF,plausibleMus,plausibleTaus) # to do: a bit nested
        return (bestMu,bestTau,posteriorPDF)
    def getPosteriorMaxLike(self,posteriorPDF,plausibleMus,plausibleTaus):
        posteriorMax = np.max(posteriorPDF.ravel())
        maxIndices = np.where(posteriorPDF==posteriorMax)
        bestMuIndex = maxIndices[0]
        bestTauIndex = maxIndices[1]
        bestMu = np.mean(plausibleMus[bestMuIndex]) # take the mean. In some cases, the ML can span two (closely spaced) points, particularly early on/when broad. So go for the middle.
        bestTau = np.mean(plausibleTaus[bestTauIndex])
        return (bestMu,bestTau)

    def makeFigure(self,plausibleMus,plausibleTaus,theta):
        # get data
        mu,_lambda,alpha,beta = self.unpackTheta(theta)   
        bestMu,bestTau,posteriorPDF = normalConjugate.getPosteriorStats(plausibleMus,plausibleTaus,theta)
        bestSigma = 1/np.sqrt(bestTau)
        sourceX = np.linspace(-4*bestSigma,4*bestSigma,200) + bestMu
        estimatePDF = scipy.stats.norm.pdf(sourceX,bestMu,bestSigma)
        # make figure
        fig = plt.figure(figsize=plt.figaspect(.5))  
        axSourceEstimate = fig.add_subplot(1,2,1)
        axPosterior = fig.add_subplot(1,2,2)
        # plot estimate pdf
        estimateLine, = axSourceEstimate.plot(sourceX,estimatePDF)
        estimateLine.set_linewidth(3)
        estimateLine.set_color((0.,0.,1.))#'blue')       
        axSourceEstimate.set_title('estimate')
        axSourceEstimate.set_ylim(0,axSourceEstimate.get_ylim()[1])
        axSourceEstimate.set_xlim(np.min(sourceX),np.max(sourceX))
        yTicks = np.round(axSourceEstimate.get_yticks(),2)
        axSourceEstimate.set_yticks([0., np.max(yTicks)])
        axSourceEstimate.set_xlabel('data units')
        axSourceEstimate.set_ylabel('density')
        axSourceEstimate.spines['top'].set_visible(False)
        axSourceEstimate.spines['right'].set_visible(False)
        # plot posterior
        axPosterior.imshow(posteriorPDF.T)
        axPosterior.invert_yaxis()
        xTickindices = np.linspace(0,np.shape(plausibleMus)[0] - 1,5).astype(int)
        muTickLabels = plausibleMus[xTickindices]
        axPosterior.set_xticks(xTickindices)       
        axPosterior.set_xticklabels(np.round(muTickLabels,1))
        yTickIndices = np.linspace(0,np.shape(plausibleTaus)[0] - 1,5).astype(int)
        tauTickLabels = plausibleTaus[yTickIndices]
        axPosterior.set_yticks(yTickIndices)
        axPosterior.set_yticklabels(np.round(tauTickLabels,1))     
        axPosterior.set_xlabel(r'$\mu$')
        axPosterior.set_ylabel(r'$\tau$') 
        titleString = 'posterior: ' + r'$\theta$' + ' = ' + '[' + str(round(mu,1)) + ', ' + str(round(_lambda,1)) + ', ' + str(round(alpha,1)) + ', ' + str(round(beta,1)) + ']'
        axPosterior.set_title(titleString)
        # adjust positions
        axHeight = .855
        axWidth = .42
        colSpace = .075
        pos = [.07,.1,axWidth,axHeight]
        axSourceEstimate.set_position(pos,which = 'both')
        pos = copy.deepcopy(pos) # unsure if the copy will ever matter but just in case
        pos[0] = pos[0] + axWidth + colSpace
        axPosterior.set_position(pos)
        return (fig,axSourceEstimate,axPosterior,estimateLine)
    def addTrueSourcePlots(self,axSourceEstimate,estimateLine,mu,sigma):
        # plot source
        xlim = axSourceEstimate.get_xlim()
        sourceX = np.linspace(xlim[0],xlim[1],200)
        sourcePDF = scipy.stats.norm.pdf(sourceX,mu,sigma)
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

fileName = 'testNormal_DataStruct.mat'
floatPrecisionTolerance = 1e-10 # see loop below. allow for trivial deviations between matlab/python output related to factors such as double-precison, minor differences in distribution implementations, etc.

path = os.path.dirname(os.path.realpath(__file__))
fullFileName = os.path.join(path,fileName)

dataStruct = scipy.io.loadmat(fullFileName)
dataStruct = dataStruct['dataStruct'] # should be a structured np array with fields corresponding to those created in matlab (datasets,bestMus,bestTaus,and finalThetas)

datasets = dataStruct[0][0]['datasets'] # rows = datsets / cols = observations
matlabMus = dataStruct[0][0]['bestMus']
matlabTaus = dataStruct[0][0]['bestTaus']
matlabThetas = dataStruct[0][0]['finalThetas']

normalConjugate = Normal()
numEvaluationPoints = 20 # precision level for evaluating tau and theta
for dataI in range(np.shape(datasets)[0]):
    currDataset = datasets[dataI,:]
    mu = dataI+1 # see testNormal_part1 for explanation
    sigma = 2
    plausibleMus = np.linspace(-4*sigma,4*sigma,numEvaluationPoints) + mu
    plausibleTaus = np.linspace(.001,(1/(sigma**2))*4.5,numEvaluationPoints)
    pyTheta = normalConjugate.jeffreysPrior
    for datum in currDataset:
        pyTheta = normalConjugate.updateTheta(datum,pyTheta)

    pyMu,pyTau,_ = normalConjugate.getPosteriorStats(plausibleMus,plausibleTaus,pyTheta)
    matlabTheta = matlabThetas[dataI,:]
    matlabMu = matlabMus[dataI]
    matlabTau = matlabTaus[dataI]

    print(' ')
    print('dataset ' + str(dataI+1))
    print('difference theta',np.sum(matlabTheta - pyTheta))
    print('difference mu',matlabMu - pyMu)
    print('difference tau',matlabTau - pyTau)

    if np.sum(matlabTheta - pyTheta) != 0: # should be exact match
        print('pyTheta',pyTheta)
        print('matlabTheta',matlabTheta)
        raise Exception('unacceptable deviation between matlab and python detected for theta. See output above.')
    if matlabMu - pyMu > floatPrecisionTolerance:
        print('pyMu',pyMu)
        print('matlabMu',matlabMu)
        raise Exception('unacceptable deviation between matlab and python detected for mu. See output above.')
    if matlabTau - pyTau > floatPrecisionTolerance:
        print('pyTau',pyTau)
        print('matlabTau',matlabTau)
        raise Exception('unacceptable deviation between matlab and python detected for tau. See output above.')

    









