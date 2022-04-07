#from cmath import nan
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import copy

# to do: check kl divergence implementation
# mu of posterior seems to not be affected by positive skew. typically to the left of the mode but should be to the right?
# ok and the MLE of the posterior is definitely not the ML of the posterior. maybe check with Randy.
# make figure summary colors consistent across classes
# handle multiple data inputs for theta function?
# check why passing in last data value when getting posterior stats. should just update and query based on theta alone?
# in figure, set muX to posteriorX / same for everything just allowing for the other case
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

print('Generating data')
mu = .2
numDataPoints = 50
dataPoints = np.random.exponential(scale = mu, size = (numDataPoints,))

print('Making exponential object')
exponentialConjugate = Exponential()

print('Setting analysis parameters')
theta = np.array([.5,0]) # Jeffreys prior 
numPosteriorBins = 500 # number of points to evaluate when computing posterior cdf/pdf

print('Running analysis')
cumulativeDataPoints = np.cumsum(dataPoints)
for dataI,dataValue in enumerate(cumulativeDataPoints):
    n = dataI+1
    theta = exponentialConjugate.updateTheta(n,dataValue,theta)

bestLambda,maxLikeLambda,posteriorPDF,lambdaX,muX = exponentialConjugate.getPosteriorStats(n,dataValue,theta,numPosteriorBins)

print('Making analysis output figure')
fig,axSourceEstimate,axPosterior,estimateLine = exponentialConjugate.makeFigure(n,dataValue,theta,numPosteriorBins)
print('Adding true parameters to figure')
exponentialConjugate.addTrueSourcePlots(axSourceEstimate,estimateLine,mu)

print('Finished. Best parameters are:')
print('lambda',bestLambda)
print('mu', 1/bestLambda)

plt.show()