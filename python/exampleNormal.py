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
mu = 2
sigma = .2
dataPoints = np.random.normal(loc = mu, scale = sigma, size = (30,))
  
print('Setting prior')
normalConjugate = Normal()
theta = normalConjugate.jeffreysPrior # for reference: [0, 0, -.5, 0] 

print('Updating theta based on data')
for dataI,dataValue in enumerate(dataPoints):
    theta = normalConjugate.updateTheta(dataValue,theta)

print('Getting stats')
plausibleMus = np.linspace(-4*sigma,4*sigma,500) + mu
plausibleTaus = np.linspace(.001,(1/(sigma**2))*4.5,500)
bestMu,bestTau,posteriorPDF = normalConjugate.getPosteriorStats(plausibleMus,plausibleTaus,theta)

print('Generating summary figure')
fig,axSourceEstimate,axPosterior = normalConjugate.initSummaryFigure()
estimateLine = normalConjugate.plotAnalysisOutput(axSourceEstimate,axPosterior,plausibleMus,plausibleTaus,theta)
normalConjugate.plotSimulatedData(axSourceEstimate,estimateLine,mu,sigma)

print('Finished. Best parameters are:')
print('mu', bestMu)
print('sigma',1/np.sqrt(bestTau))

plt.show()