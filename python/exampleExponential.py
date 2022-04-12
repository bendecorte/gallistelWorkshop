# illustrates usage of gamma conjugate prior to estimate parameters of an exponential source distribution

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