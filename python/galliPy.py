# Code by Benjamin John De Corte (i.e., bjd). If code doesn't work, send nasty message to bendecorte on github.
# Description:
# Contains all relevant code for using conjugate priors for the Bernoulli, Exponential, and Normal distributions, as well as computing computing Kullback-Leibler divergence 
# measures. This code was intitially intended to accompany a workshop given by CR Gallistel in 2022 for applying these analyses to conditioning data.
# Important notes:
# - While code is in development, no full-fledged package at this point. So importing is very basic and (therefore) you'll want to keep track of the directory containing galliPy.py.
# - Each class has example codes illustrating use that should be in this directory if pulling from github. Those are titled example{RelevantDistribution}.py
# - some error-checking on inputs is incorporated, but it will not be air-tight. In general, stick with 1-d numpy arrays whenever possible and things should function properly. 

# General class schema (for distribution-specific info. See notes in the class itself):
# Relevant functions:
# 1. updateTheta(): Updates the parameter vector for the corresponding conjugate distribution
# 2. getPosteriorStats(): Returns relevant estimates of the true underlying distribution based on the data thus far
# 3. makeFigure(): Makes a figure showing the data-estimates based on the data thus far.
# 4. addTrueSourcePlots(): When simulating data, you can also add the 'ground truth' data to the summary figure using thisi function
# 5. getKLDivergence() (or getKLDivergenceProbability for the Bernoulli dist.): Returns estimates of KL divergence for a given distribution. 
# Relevant properties:
# 1. jeffreysPrior: Just stores the jeffreys prior so you don't have to keep looking that up. 

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

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import copy

# minor to do's:
# overall:
# -better documentation for inputs/outputs for each class
# exponential class:
# -reduce inputs needed for the getPosteriorStats function. could store the last n/data point as a property. keeping consistent with the matlab code for now. 
# -make figure summary colors consistent across classes
# -multiple data inputs for theta function?
# bernoulli class:
# -add function that will do initial scan of the data before loop to allow user to verify in correct format (i.e., all 0's and 1's)? Could maybe do for the other classes too, but weirder with floats
# normal class:
# maybe add a method for getting default values for mu/sigma based on user data

class Bernoulli:
    def __init__(self):
        self.jeffreysPrior = np.array([.5,.5])    
    def updateTheta(self,data,thetaCurrent):
        # updates the hyperparameters for the normal-gamma distribution.
        # inputs: 1 = numpy float/array of data (ideally, logicals, but also floats that are either 0 or 1), 2 = current theta as a 2-parameter np array
        # outputs: the updated theta as a 1-d numpy array
        self.checkUpdateInputs(data,thetaCurrent)
        alphaCurrent,betaCurrent = self.unpackTheta(thetaCurrent)
        numSuccesses = np.sum(data==1)
        numFailures = np.sum(data==0)
        alphaNew = alphaCurrent + numSuccesses
        betaNew = betaCurrent + numFailures
        thetaNew = np.array([alphaNew,betaNew])
        return thetaNew
    def checkUpdateInputs(self,data,thetaCurrent):
        # very basic error checking.
        if not data.size==1: # the ideal input is single value (can be float, logical, etc. though)
            if not isinstance(data, np.ndarray) or data.ndim!=1 and data.size>1: # basic check on array inputs
                raise Exception('data must be a numpy float or 1-d numpy array')
        if not isinstance(thetaCurrent, np.ndarray) or thetaCurrent.ndim!=1:
            raise Exception('theta must be 1-d numpy array')
    def unpackTheta(self,thetaCurrent):
        # convenience function for making code more explicit. 
        alpha = thetaCurrent[0]
        beta = thetaCurrent[1]
        return(alpha,beta)
    def getPosteriorStats(self,theta):
        # get estimates from posterior based on current theta
        # inputs: current theta as a 1-d numpy array
        # outputs: tuple with the best estimate of the Bernoulli p parameter, the posterior, and the corresponding x-values for the posterior (mainly for plotting purposes in this case), respectively.  
        alpha,beta = self.unpackTheta(theta)   
        posteriorX = np.linspace(.005,.995,100)
        posteriorPDF = scipy.stats.beta.pdf(posteriorX,alpha,beta)
        posteriorPDF = posteriorPDF/np.sum(posteriorPDF)
        bestP = np.sum(posteriorX*posteriorPDF)
        return (bestP,posteriorPDF,posteriorX)
    def getJointPosterior(self,theta1,comparison):
        # gets estimates of joint posterior for 1 Bernoulli distribution relative to some null-value or another bernolli distribution
        # inputs: 1 = theta for the primary distribution of interest. 2. Either a single value between 0 and 1 (non-inclusive) giving a null comparison value or another theta distribution as a 1-d numpy array
        # outputs: tuple with the joint posterior, first posterior pdf, and second posterior pdf, respectively.
        _,posteriorPDF1,posteriorX = self.getPosteriorStats(theta1)
        comparison = self.checkJointComparisonInputs(comparison)
        if comparison.size==1: # comparing to a specified null value
            nullValueIndex = np.min(np.where(posteriorX>comparison))
            posteriorPDF2 = np.zeros(np.shape(posteriorPDF1))
            posteriorPDF2[nullValueIndex] = 1 # all of the mass gets put at the point corresponding to the null value
        if comparison.size==2: # comparing to a second distribution-estimate
            _,posteriorPDF2,_ = self.getPosteriorStats(comparison)
        posterior1Grid,posterior2Grid = np.meshgrid(posteriorPDF1,posteriorPDF2,indexing = 'ij')        
        jointPosterior = posterior1Grid*posterior2Grid
        return jointPosterior,posteriorPDF1,posteriorPDF2
    def checkJointComparisonInputs(self,comparison):
        # basic error checking
        if isinstance(comparison,float): # in particular, looking to see if they passed in a python-float. numpy floats would be fine, but below conversion doesn't hurt.  
            comparison = np.array(comparison) # not necessary, but keeps the main function's code clearer with the .size
        if comparison.size==1:
            if comparison<=0 or comparison>=1:
                raise Exception('null comparison value must be between 0 and 1 (non-inclusive)')
        if comparison.size>2:
            raise Exception('comparison inputs to getJointPosterior() must either be a bernoulli theta (2-values) or a null comparison probability (1-value)')
        return comparison
    def getKLDivergenceProbability(self,n,k,p):
        # gets PROBABILITY VALUE of kl divergence between two distributions. NOT the KL divergence itself in this case. See manunscript for conversion.
        # inputs: 1. the number of data points, 2. the number of successful draws, 3. the p-parameter to test against the data (1-d numpy array or single-value)
        n,k,p = self.checkKLInputs(n,k,p)
        # probability of nDkl
        probability = p**k*(1-p)**(n-k)*scipy.special.factorial(n)/(scipy.special.factorial(k)*scipy.special.factorial(n-k))
        # prob greater than (or equal to)
        kk = copy.deepcopy(k)
        probabilityGreater = copy.deepcopy(probability)
        for i in np.arange(kk[0],n[0],1):
            kk = kk+1
            probabilityGreater = probabilityGreater + p**kk*(1-p)**(n-kk)*scipy.special.factorial(n)/(scipy.special.factorial(kk)*scipy.special.factorial(n-kk))
        # prob less than (or equal to)
        kk = copy.deepcopy(k)
        probabilityLess = copy.deepcopy(probability)
        for i in np.arange(kk[0],0,-1):
            kk = kk-1
            probabilityLess = probabilityLess + p**kk*(1-p)**(n-kk)*scipy.special.factorial(n)/(scipy.special.factorial(kk)*scipy.special.factorial(n-kk))            
        return probability,probabilityLess,probabilityGreater
    def checkKLInputs(self,n,k,p):
        # basic error checking. note that, even when a float is given, converting to 1-d numpy array makes things easier to handle. 
        if len(np.shape(p))!=0: # if requesting probability of more than one value
            n = np.ones(np.shape(p)) * n
            k = np.ones(np.shape(p)) * k    
        if len(np.shape(n))==0:
            n = np.array(n)
            n = np.reshape(n,(1,))
        if len(np.shape(k))==0:
            k = np.array(k)
            k = np.reshape(k,(1,))
            print('k',k)
        if len(np.shape(p))==0:
            p = np.array(p)
            p = np.reshape(p,(1,))
        return n,k,p    
    def makeFigure(self,theta):
        # makes a summary figure for the current theta
        # get data
        alpha,beta = self.unpackTheta(theta) 
        bestP,posteriorPDF,posteriorX = self.getPosteriorStats(theta)
        # make figure
        fig = plt.figure(figsize=plt.figaspect(.5))  
        axSourceEstimate = fig.add_subplot(1,2,1)
        axPosterior = fig.add_subplot(1,2,2)
        # plot estimate pdf
        barWidth = .4
        estimateBarFail, = axSourceEstimate.bar(0.,1-bestP,width = barWidth,color = [0,1,0])
        estimateBarSuccess, = axSourceEstimate.bar(1.,bestP,width = barWidth,color = [0,1,0])
        axSourceEstimate.set_title('estimate')
        axSourceEstimate.set_xlim(-.5,1.5)
        axSourceEstimate.set_xticks([0.,1.])
        axSourceEstimate.set_xticklabels(('failure','success'))
        axSourceEstimate.set_ylim(0,1)
        axSourceEstimate.set_yticks([0,.25,.5,.75,1])
        axSourceEstimate.set_ylabel('probability')
        # plot posterior
        posteriorLine, = axPosterior.plot(posteriorX,posteriorPDF)
        posteriorLine.set_linewidth(3)
        posteriorLine.set_color('red')
        bestPLine, = axPosterior.plot([bestP,bestP],axPosterior.get_ylim())
        bestPLine.set_linewidth(3)
        bestPLine.set_color([0,1,0])
        axPosterior.set_xlim(np.min(posteriorX),np.max(posteriorX))
        axPosterior.set_xlabel(r'$\itp$')
        titleString = 'posterior: ' + r'$\theta$' + ' = ' + '[' + str(round(alpha,1)) + ', ' + str(round(beta,1)) + ']'
        axPosterior.set_title(titleString)
        yTicks = np.round(axPosterior.get_yticks(),3)
        axPosterior.set_yticks([0., np.max(yTicks)])
        # a few asthetics applied to both axes
        axisList = [axSourceEstimate,axPosterior]
        for axCurr in axisList:
            axCurr.spines['top'].set_visible(False)
            axCurr.spines['right'].set_visible(False)
            axCurr.set_ylim(0,axCurr.get_ylim()[1])
        bestPLine.set_ydata(axPosterior.get_ylim())
        # adjust positions
        axHeight = .855
        axWidth = .42
        colSpace = .075
        pos = [.07,.095,axWidth,axHeight]
        axSourceEstimate.set_position(pos,which = 'both')
        pos = copy.deepcopy(pos) # unsure if the copy will ever matter but just in case
        pos[0] = pos[0] + axWidth + colSpace
        axPosterior.set_position(pos)
        return (fig,axSourceEstimate,axPosterior,estimateBarSuccess,estimateBarFail)
    def addTrueSourcePlots(self,axSourceEstimate,trueP,estimateBarSuccess,estimateBarFail):
        # adds 'ground truth' data for explicitly simulated Bernoulli distribution. 
        # get existing bar info
        barWidth = estimateBarSuccess.get_width()
        failX = estimateBarFail.get_x()
        successX = estimateBarSuccess.get_x()       
        #right justify
        estimateBarFail.set_x(failX + (barWidth/2))
        estimateBarSuccess.set_x(successX + (barWidth/2))       
        #add source bars
        trueBarFail, = axSourceEstimate.bar(failX,1-trueP,width = barWidth,color = [0,0,0])
        trueBarSuccess, = axSourceEstimate.bar(successX,trueP,width = barWidth,color = [0,0,0])
        # update legend/text
        axSourceEstimate.set_title(' ') # want to keep the space occupied by the legend, so don't delete
        textX = axSourceEstimate.get_xlim()[0] + (.1*np.diff(axSourceEstimate.get_xlim()))
        textY = axSourceEstimate.get_ylim()[1] * 1.0125
        estimateText = axSourceEstimate.text(textX,textY,'estimate')
        estimateText.set_color(estimateBarSuccess.get_facecolor())
        textX = copy.deepcopy(textX) + (.25*np.diff(axSourceEstimate.get_xlim()))
        textY = copy.deepcopy(textY)
        trueText = axSourceEstimate.text(textX,textY,'true')
        trueText.set_color(trueBarSuccess.get_facecolor())

class Exponential:
    def __init__(self):
        self.jeffreysPrior = np.array([.5,0])    
    def updateTheta(self,n,data,thetaCurrent):
        # updates the hyperparameters for the gamma distribution.
        # inputs: 1 = cumulative sample #, 2 = data sample value, 3. current theta
        # outputs: the updated theta as a 1-d numpy array
        self.checkUpdateInputs(data,thetaCurrent)
        alphaCurrent,betaCurrent = self.unpackTheta(thetaCurrent)
        alphaNew = alphaCurrent + n
        betaNew = betaCurrent + data
        thetaNew = np.array([alphaNew,betaNew])
        return thetaNew
    def checkUpdateInputs(self,data,thetaCurrent):
        # basic error-checking
        if not isinstance(data, np.floating): # the ideal input
            if not isinstance(data, np.ndarray) or data.ndim!=1 and data.size>1: # basic check on array inputs
                raise Exception('data must be a numpy float or 1-d numpy array')
        if not isinstance(thetaCurrent, np.ndarray) or thetaCurrent.ndim!=1:
            raise Exception('theta must be 1-d numpy array')
    def unpackTheta(self,thetaCurrent):
        # convenience function for making code more explicit.
        alpha = thetaCurrent[0]
        beta = thetaCurrent[1]
        return(alpha,beta)
    def getPosteriorStats(self,n,data,theta,numBins):
        # get estimates from posterior 
        # inputs: 1 = cumulative data point, 2. current data value, 3. current theta as a 1-d numpy array, 4. number of bins to use for the posterior distribution.
        # outputs: tuple with the best estimate of the Bernoulli p parameter, the posterior, and the corresponding x-values for the posterior (mainly for plotting purposes in this case), respectively.  
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
        # gets kl divergence between two distributions. 
        # inputs: 1. theta for primary distribution of interest (1-d numpy array), 2. theta for the other distribution. 
        # outputs: the divergence value
        divergence = np.log(trueDistribution) - np.log(approximatingDistribution) + (approximatingDistribution/trueDistribution) - 1
        return divergence
    def makeFigure(self,n,dataValue,theta,numPosteriorBins):
        # makes a summary figure for the current theta
        # inputs: same as self.getPosteriorStats
        # get data
        alpha,beta = self.unpackTheta(theta) 
        bestLambda,maxLikeLambda,posteriorPDF,lambdaX,muX = self.getPosteriorStats(n,dataValue,theta,numPosteriorBins)
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
        # adds 'ground truth' data for explicitly simulated Bernoulli distribution. 
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

class Normal:
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
    def getKLDivergence(self,mu1,mu2,sigma1,sigma2):
        # gets kl divergence between two distributions. 
        # inputs: 1 = mu for distribution 1, 2 = mu for distribution 2, 3 = sigma for distribution 1, 4 = sigma for distribution 2
        # outputs: the divergence value
        sigmaRatio = np.log(sigma2/sigma1)
        meanDifference = mu1 - mu2
        divergence = sigmaRatio + ( (np.square(sigma1)+np.square(meanDifference)) / (2*np.square(sigma2)) ) - .5
        return divergence
    def makeFigure(self,plausibleMus,plausibleTaus,theta):
        # generates a summary figure
        # get data
        mu,_lambda,alpha,beta = self.unpackTheta(theta)   
        bestMu,bestTau,posteriorPDF = self.getPosteriorStats(plausibleMus,plausibleTaus,theta)
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
