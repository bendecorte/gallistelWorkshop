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


class InteractiveNormalFig(Normal):
    def __init__(self,mu,sigma,numDataPoints):
        self.conjugate = Normal()
        self.theta = self.conjugate.jeffreysPrior
        self.mu = mu
        self.sigma = sigma
        self.plausibleMus = np.linspace(-4*self.sigma,4*self.sigma,500) + mu
        self.plausibleTaus = np.linspace(.001,(1/(self.sigma**2))*4.5,500)
        self.data = np.random.normal(loc = self.mu, scale = self.sigma, size = (numDataPoints,))       
        self.dataIndex = 0

        self.initializeAnimation()

    def initializeAnimation(self):
        # need to run 2 iterations to get an estimate of tau
        self.theta = self.conjugate.updateTheta(self.data[self.dataIndex],self.theta)
        self.dataIndex = self.dataIndex + 1
        self.theta = self.conjugate.updateTheta(self.data[self.dataIndex],self.theta)
        # make figure using standard funcs and connect an event function
        self.fig,self.axSourceEstimate,self.axPosterior = self.conjugate.initSummaryFigure()
        self.estimateLine = self.conjugate.plotAnalysisOutput(self.axSourceEstimate,self.axPosterior,self.plausibleMus,self.plausibleTaus,self.theta)
        self.conjugate.plotSimulatedData(self.axSourceEstimate,self.estimateLine,self.mu,self.sigma)
        self.fig.canvas.mpl_connect('button_press_event',self.updateFig)
        print('click figure to run next iteration')
        plt.show()
        # now just waiting for user to trigger the event interrupt
        

    def updateFig(self,event):
        # update theta/plot
        self.dataIndex = self.dataIndex + 1
        if self.dataIndex < np.shape(self.data)[0]:
            self.theta = self.conjugate.updateTheta(self.data[self.dataIndex],self.theta)
            self.axSourceEstimate.clear() # lazy way, but the built-in figure functions have code for adjusting axis-limits, etc, which help. 
            self.axPosterior.clear()
            self.estimateLine = self.conjugate.plotAnalysisOutput(self.axSourceEstimate,self.axPosterior,self.plausibleMus,self.plausibleTaus,self.theta)
            self.conjugate.plotSimulatedData(self.axSourceEstimate,self.estimateLine,self.mu,self.sigma)
            self.axSourceEstimate.plot(np.array([self.data[self.dataIndex],self.data[self.dataIndex]]),self.axSourceEstimate.get_ylim())
            plt.draw()
            print('click figure to run next iteration')
        else:
            print('no more data')





mu = 10
sigma = 2
numDataPoints = 500
f = InteractiveNormalFig(mu,sigma,numDataPoints)
