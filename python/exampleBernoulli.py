# illustrates usage of beta conjugate prior to estimate parameters of a Bernoulli source distribution

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
# SOFTWARE. #

import numpy as np
import matplotlib.pyplot as plt
from analysisScripts.galliPy import Bernoulli

print('Generating data')
p = .2
dataPoints = np.random.binomial(1,p, size = (50,))

print('Making bernoulli object')
bernoulliConjugate = Bernoulli()
theta = bernoulliConjugate.jeffreysPrior # for ref: np.array([.5,.5])

print('Running analysis')
for dataI,dataValue in enumerate(dataPoints):
    theta = bernoulliConjugate.updateTheta(dataValue,theta)

print('Extracting stats')
numPosteriorBins = 500 # number of points to evaluate when computing posterior cdf/pdf
bestP,posteriorPDF,posteriorX = bernoulliConjugate.getPosteriorStats(theta)

print('Making analysis output figure')
fig,axSourceEstimate,axPosterior = bernoulliConjugate.initSummaryFigure()
estimateBarSuccess,estimateBarFail = bernoulliConjugate.plotAnalysisOutput(axSourceEstimate,axPosterior,theta)
bernoulliConjugate.plotSimulatedData(axSourceEstimate,p,estimateBarSuccess,estimateBarFail)

print('Finished. Best parameter is:')
print('best probability of success',bestP)

plt.show()