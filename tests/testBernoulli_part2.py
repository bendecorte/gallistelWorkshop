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

class Bernoulli():
    def __init__(self):
        self.jeffreysPrior = np.array([.5,.5])    
    def updateTheta(self,data,thetaCurrent):
        self.checkUpdateInputs(data,thetaCurrent)
        alphaCurrent,betaCurrent = self.unpackTheta(thetaCurrent)
        numSuccesses = np.sum(data==1)
        numFailures = np.sum(data==0)
        alphaNew = alphaCurrent + numSuccesses
        betaNew = betaCurrent + numFailures
        thetaNew = np.array([alphaNew,betaNew])
        return thetaNew
    def checkUpdateInputs(self,data,thetaCurrent):
        if not data.size==1: # the ideal input is single value (can be float, logical, etc. though)
            if not isinstance(data, np.ndarray) or data.ndim!=1 and data.size>1: # basic check on array inputs
                raise Exception('data must be a numpy float or 1-d numpy array')
        if not isinstance(thetaCurrent, np.ndarray) or thetaCurrent.ndim!=1:
            raise Exception('theta must be 1-d numpy array')
    def unpackTheta(self,thetaCurrent):
        alpha = thetaCurrent[0]
        beta = thetaCurrent[1]
        return(alpha,beta)
    def getPosteriorStats(self,theta):
        alpha,beta = self.unpackTheta(theta)   
        posteriorX = np.linspace(.005,.995,100)
        posteriorPDF = scipy.stats.beta.pdf(posteriorX,alpha,beta)
        posteriorPDF = posteriorPDF/np.sum(posteriorPDF)
        bestP = np.sum(posteriorX*posteriorPDF)
        return (bestP,posteriorPDF,posteriorX)
    def getJointPosterior(self,theta1,comparison):
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
        if isinstance(comparison,float): # in particular, looking to see if they passed in a python-float. numpy floats would be fine, but below conversion doesn't hurt.  
            comparison = np.array(comparison) # not necessary, but keeps the main function's code clearer with the .size
        if comparison.size==1:
            if comparison<=0 or comparison>=1:
                raise Exception('null comparison value must be between 0 and 1 (non-inclusive)')
        if comparison.size>2:
            raise Exception('comparison inputs to getJointPosterior() must either be a bernoulli theta (2-values) or a null comparison probability (1-value)')
        return comparison
    def getKLDivergenceProbability(self,n,k,p):
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
        # get existing bar info
        barWidth = estimateBarSuccess.get_width()
        failX = estimateBarFail.get_x()
        successX = estimateBarSuccess.get_x()
        
        #right justify
        print('failX',failX)
        estimateBarFail.set_x(failX + (barWidth/2))
        estimateBarSuccess.set_x(successX + (barWidth/2))
        
        #add source bars
        print('failX',failX)
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

fileName = 'testBernoulli_DataStruct.mat'
floatPrecisionTolerance = 1e-10 # see loop below. allow for trivial deviations between matlab/python output related to factors such as double-precison, minor differences in distribution implementations, etc.

path = os.path.dirname(os.path.realpath(__file__))
fullFileName = os.path.join(path,fileName)

dataStruct = scipy.io.loadmat(fullFileName)
dataStruct = dataStruct['dataStruct'] # should be a structured np array with fields corresponding to those created in matlab (datasets,bestMus,bestTaus,and finalThetas)

datasets = dataStruct[0][0]['datasets'] # rows = datsets / cols = observations
matlabPs = dataStruct[0][0]['ps']
matlabThetas = dataStruct[0][0]['finalThetas']
matlabNullJoints = dataStruct[0][0]['nullJoints']
matlabDualJoints = dataStruct[0][0]['dualJoints']

bernoulliConjugate = Bernoulli()
# basic ps
for dataI in range(np.shape(datasets)[0]):
    currDataset = datasets[dataI,:]
    pyTheta = bernoulliConjugate.jeffreysPrior
    for datumI,datum in enumerate(currDataset):
        pyTheta = bernoulliConjugate.updateTheta(datum,pyTheta)

    pyP,_,_ = bernoulliConjugate.getPosteriorStats(pyTheta)

    matlabTheta = matlabThetas[dataI,:]
    matlabP = matlabPs[dataI,0]

    print(' ')
    print('dataset ' + str(dataI+1))
    print('difference theta',np.sum(matlabTheta - pyTheta))
    print('difference p',matlabP - pyP)

    if np.sum(matlabTheta - pyTheta) != 0: # should be exact match
        print('pyTheta',pyTheta)
        print('matlabTheta',matlabTheta)
        raise Exception('unacceptable deviation between matlab and python detected for theta. See output above.')
    if matlabP - pyP > floatPrecisionTolerance:
        print('pyP',pyP)
        print('matlabP',matlabP)
        raise Exception('unacceptable deviation between matlab and python detected for p. See output above.')

# null p joint distributions
nullValue = .2
for dataI in range(np.shape(datasets)[0]):
    currDataset = datasets[dataI,:]
    pyTheta = bernoulliConjugate.jeffreysPrior
    for datumI,datum in enumerate(currDataset):
        pyTheta = bernoulliConjugate.updateTheta(datum,pyTheta)

    pyNull,_,_ = bernoulliConjugate.getJointPosterior(pyTheta,nullValue)

    nullI = matlabNullJoints[:,0]==(dataI+1)
    matlabNull = matlabNullJoints[nullI,1:] # first row is the dummy of the dataset

    difference = matlabNull - pyNull

    print(' ')
    print('dataset ' + str(dataI+1))
    print('summed difference null',np.sum(difference.ravel()))

    if np.max(difference.ravel()) > floatPrecisionTolerance:
        print('pyP',pyP)
        print('matlabP',matlabP)
        raise Exception('unacceptable deviation between matlab and python detected for null joint dists. See output above.')
    
# dual joint distributions
bernoulli1 = Bernoulli()
bernoulli2 = Bernoulli()
dataIndices = np.arange(0,np.shape(datasets)[0])
for dataI in range(np.shape(dataIndices)[0]-1):
    currData1 = datasets[dataI,:]
    currData2 = datasets[dataI+1,:]

    theta1 = bernoulli1.updateTheta(currData1,bernoulli1.jeffreysPrior)
    theta2 = bernoulli1.updateTheta(currData2,bernoulli1.jeffreysPrior)

    pyDual,_,_ = bernoulliConjugate.getJointPosterior(theta1,theta2)

    dualI = matlabDualJoints[:,0]==(dataI+1)
    matlabDual = matlabDualJoints[dualI,1:] # first row is the dummy of the dataset

    difference = matlabDual - pyDual  

    print(' ')
    print('dataset ' + str(dataI+1))
    print('summed difference dual',np.sum(difference.ravel()))

    if np.max(difference.ravel()) > floatPrecisionTolerance:
        print('pyDual',pyDual)
        print('matlabDual',matlabDual)
        raise Exception('unacceptable deviation between matlab and python detected for dual joint dists. See output above.')
    







