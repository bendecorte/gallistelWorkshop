import numpy as np
import matplotlib.pyplot as plt
from galliPy import Bernoulli


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