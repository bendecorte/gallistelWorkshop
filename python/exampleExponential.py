import numpy as np
import matplotlib.pyplot as plt
from galliPy import Exponential

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