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
from analysisScripts.galliPy import Exponential


class InteractiveExponentialFig(Exponential):
    def __init__(self,mu,numDataPoints):
        self.conjugate = Exponential()
        self.theta = self.conjugate.jeffreysPrior
        self.mu = mu
        self.data = np.random.exponential(scale = mu, size = (numDataPoints,))
        self.data = np.cumsum(self.data)
        self.dataIndex = 0

        self.initializeAnimation()

    def initializeAnimation(self):
        # do first iteration
        self.theta = self.conjugate.updateTheta(self.dataIndex+1,self.data[self.dataIndex],self.theta) # n, datum, and theta, respectively
        # make figure using standard funcs and connect an event function
        self.fig,self.axSourceEstimate,self.axPosterior = self.conjugate.initSummaryFigure()
        self.estimateLine = self.conjugate.makeFigure(self.axSourceEstimate,self.axPosterior,self.dataIndex + 1,self.data[self.dataIndex],self.theta,500) # last ones: n, datum, theta, and number of bins for posterior, respectively
        self.conjugate.addTrueSourcePlots(self.axSourceEstimate,self.estimateLine,self.mu)
        self.fig.canvas.mpl_connect('button_press_event',self.updateFig)
        print('click figure to run next iteration')
        plt.show()
        # now just waiting for user to trigger the event interrupt
        
    def updateFig(self,event):
        # update theta/plot
        self.dataIndex = self.dataIndex + 1
        if self.dataIndex < np.shape(self.data)[0]:
            self.theta = self.conjugate.updateTheta(self.dataIndex+1,self.data[self.dataIndex],self.theta) # n, datum, and theta, respectively
            self.axSourceEstimate.clear() # lazy way, but the built-in figure functions have code for adjusting axis-limits, etc, which help. 
            self.axPosterior.clear()
            self.estimateLine = self.conjugate.makeFigure(self.axSourceEstimate,self.axPosterior,self.dataIndex + 1,self.data[self.dataIndex],self.theta,500)
            self.conjugate.addTrueSourcePlots(self.axSourceEstimate,self.estimateLine,self.mu)
            newValue = self.data[self.dataIndex] - self.data[self.dataIndex-1] # keep in mind the data are cumulative, so subtract with prior value to get the actual added value. 
            self.axSourceEstimate.plot(np.array([newValue,newValue]),self.axSourceEstimate.get_ylim())
            plt.draw()
            print('click figure to run next iteration')
        else:
            print('no more data')





mu = .2
numDataPoints = 500
f = InteractiveExponentialFig(mu,numDataPoints)
