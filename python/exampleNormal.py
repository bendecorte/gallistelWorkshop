# BJD. Code for estimating the hyperparameters of the normalgamma prior on the mean and precision of the normal distribution, including a simple illustration with simulated data.
# The Normal() class contains all relevant methods for the analysis. The most relevant ones are:
# 1. updateTheta(data,theta): Returns the updated hyperparameters based on the incoming data and current hyperparameter values (theta)
# 2. getPosteriorStats(mus,taus,theta): Evaluates the normalgamma distribution a points for mu and tau specified by the user (mus and taus, respectively), based on 
#                                       the current hyperparameters (theta). 
#                                       Returns the best estimate of the mean and precision (maximum likelihood estimates) and also the pdf.       
# For further illustration on use, see the simple simulation below the class.   
# In general, all inputs should be 1-d numpy arrays or floats. Some basic checks are included, but they are not necessarily exhaustive.               

import numpy as np
import matplotlib.pyplot as plt
from galliPy import Normal

print('Generating data')
mu = 10
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