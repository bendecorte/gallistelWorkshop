# BJD. Code for estimating the hyperparameters of the normalgamma prior on the mean and precision of the normal distribution, including a simple illustration with simulated data.
# The Normal() class contains all relevant methods for the analysis. The most relevant ones are:
# 1. updateTheta(data,theta): Returns the updated hyperparameters based on the incoming data and current hyperparameter values (theta)
# 2. getPosteriorStats(mus,taus,theta): Evaluates the normalgamma distribution a points for mu and tau specified by the user (mus and taus, respectively), based on 
#                                       the current hyperparameters (theta). 
#                                       Returns the best estimate of the mean and precision (maximum likelihood estimates) and also the pdf.       
# For further illustration on use, see the simple simulation below the class.   
# In general, all inputs should be 1-d numpy arrays or floats. Some basic checks are included, but they are not necessarily exhaustive.               

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import copy

class Normal():
    def __init__(self):
        # note: if faster runtime needed. Could store other params inside object at startup (e.g., plausibleMus, grids, etc). The functions here will regenerate things like the mu/tau grids on the fly for flexibility/readability, which wouldn't be required. 
        self.jeffreysPrior = np.array([0, 0, -.5, 0])    
    def updateTheta(self,data,thetaCurrent):
        # updates the hyperparameters. data: incoming data is a numpy float or a 1-d numpy array. thetaCurrent: current hyperparameter estimates as a 1-d numpy array.
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
        # basic checks on the inputs to updateTheta
        if not isinstance(data, np.floating): # the ideal input
            if not isinstance(data, np.ndarray) or data.ndim!=1 and data.size>1: # basic check on array inputs
                raise Exception('data must be a numpy float or 1-d numpy array')
        if not isinstance(thetaCurrent, np.ndarray) or thetaCurrent.ndim!=1:
            raise Exception('theta must be 1-d numpy array')
    def unpackTheta(self,thetaCurrent):
        # convenience method for breaking apart the hyperparameter vector (theta) into component variables. 
        mu = thetaCurrent[0]
        _lambda = thetaCurrent[1] # reserved in python
        alpha = thetaCurrent[2]
        beta = thetaCurrent[3]
        return (mu,_lambda,alpha,beta)
    def getPosteriorStats(self,plausibleMus,plausibleTaus,theta):
        # evaluates the normalgamma pdf and estracts estimates of the mean and tau. plausibleMus/Taus are specified by the user and compose the x/y axes for evaluating the pdf. theta is the hyperparameter vector for the normalgamma pdf.
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
        # extracts maximum likelihood estimates for mu and tau from the normalgamma pdf. posteriorpdf is a 2-d numpy array containing the normalgamma prior/posterior. plausibleMus/Taus are 1-d numpy arrays that correspond to the rows and columns of the pdf, respectivley. 
        posteriorMax = np.max(posteriorPDF.ravel())
        maxIndices = np.where(posteriorPDF==posteriorMax)
        bestMuIndex = maxIndices[0]
        bestTauIndex = maxIndices[1]
        bestMu = np.mean(plausibleMus[bestMuIndex]) # take the mean. In some cases, the ML can span two (closely spaced) points, particularly early on/when broad. So go for the middle.
        bestTau = np.mean(plausibleTaus[bestTauIndex])
        return (bestMu,bestTau)
    def makeFigure(self,plausibleMus,plausibleTaus,theta):
        # generates a summary figure
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
        # adds summary of 'ground-truth' parameters when running the analysis on simulated data. axSourceEstimate and estimateLine come from the initial makeFigure method. mu and sigma are the normal parameters specified by the user for a simulation. 
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

print('Generating data')
mu = 2
sigma = 2
numDataPoints = 20
dataPoints = np.random.normal(loc = mu, scale = sigma, size = (numDataPoints,))
theta = np.array([0, 0, -.5, 0]) # Jeffreys prior. Note that, for convenience, this is also stored on init as a class property, which can be called here instead if desired.  

print('Making normal-gamma object')
normalConjugate = Normal()

print('Setting analysis parameters')
numEvaluationPoints = 500 # number of points to evaluate posterior between range for each parameter defined below
plausibleMus = np.linspace(-4*sigma,4*sigma,numEvaluationPoints) + mu
plausibleTaus = np.linspace(.001,(1/(sigma**2))*4.5,numEvaluationPoints)

print('Running analysis')
for dataI,dataValue in enumerate(dataPoints):
    theta = normalConjugate.updateTheta(dataValue,theta)
    bestMu,bestTau,posteriorPDF = normalConjugate.getPosteriorStats(plausibleMus,plausibleTaus,theta)

print('Making analysis output figure')
fig,axSourceEstimate,axPosterior,estimateLine = normalConjugate.makeFigure(plausibleMus,plausibleTaus,theta)

print('Adding true parameters to figure')
normalConjugate.addTrueSourcePlots(axSourceEstimate,estimateLine,mu,sigma)

print('Finished. Best parameters are:')
print('mu', bestMu)
print('sigma',1/np.sqrt(bestTau))

plt.show()