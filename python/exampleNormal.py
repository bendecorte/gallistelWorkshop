# illustrates usage of normalgamma conjugate prior to estimate parameters of a normal source distribution

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
from analysisScripts.galliPy import Normal

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

fig,axSourceEstimate,axPosterior = normalConjugate.initSummaryFigure()
estimateLine = normalConjugate.makeFigure(axSourceEstimate,axPosterior,plausibleMus,plausibleTaus,theta)

print('Adding true parameters to figure')
normalConjugate.addTrueSourcePlots(axSourceEstimate,estimateLine,mu,sigma)

print('Finished. Best parameters are:')
print('mu', bestMu)
print('sigma',1/np.sqrt(bestTau))

plt.show()