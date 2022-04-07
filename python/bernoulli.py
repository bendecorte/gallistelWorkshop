import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import copy

# left off: realigning bars when adding source data. looking at bar.set_x()
# to do: 
# potential bug in randy code for bernoulli kl divergence. in while loops where multiple ps are requested, wouldn't kk<n return multiple values. Need to check
# add function that will do initial scan of the data before loop to allow user to verify in correct format (i.e., all 0's and 1's)? Could maybe do for the other classes too, but weirder with floats
# code 2-distribution case (use a separate function though)
# adaptable or built-in beta x values on init?
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

print('Generating data')
p = .2
numDataPoints = 50
dataPoints = np.random.binomial(1,p, size = (numDataPoints,))

print('Making bernoulli object')
bernoulliConjugate = Bernoulli()

print('Setting analysis parameters')
theta = np.array([.5,.5]) # Jeffreys prior 
numPosteriorBins = 500 # number of points to evaluate when computing posterior cdf/pdf

print('Running analysis')
for dataI,dataValue in enumerate(dataPoints):
    theta = bernoulliConjugate.updateTheta(dataValue,theta)

bestP,posteriorPDF,posteriorX = bernoulliConjugate.getPosteriorStats(theta)

print('Making analysis output figure')
fig,axSourceEstimate,axPosterior,estimateBarSuccess,estimateBarFail = bernoulliConjugate.makeFigure(theta)

print('Adding true parameters to figure')
bernoulliConjugate.addTrueSourcePlots(axSourceEstimate,p,estimateBarSuccess,estimateBarFail)

bestP,_,_ = bernoulliConjugate.getPosteriorStats(theta)

print('Finished. Best parameter is:')
print('best probability of success',bestP)

plt.show()