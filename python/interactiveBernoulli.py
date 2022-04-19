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
from analysisScripts.galliPy import Bernoulli


class InteractiveBernoulliFig(Bernoulli):
    def __init__(self,p,numDataPoints):
        self.conjugate = Bernoulli()
        self.theta = self.conjugate.jeffreysPrior
        self.p = p
        self.data = np.random.binomial(1,p, size = (numDataPoints,))
        self.dataIndex = 0

        self.initializeAnimation()

    def initializeAnimation(self):
        # do first iteration
        self.theta = self.conjugate.updateTheta(self.data[self.dataIndex],self.theta) 
        # make figure using standard funcs and connect an event function
        self.fig,self.axSourceEstimate,self.axPosterior = self.conjugate.initSummaryFigure()
        self.estimateBarSuccess,self.estimateBarFail = self.conjugate.plotAnalysisOutput(self.axSourceEstimate,self.axPosterior,self.theta)
        self.conjugate.plotSimulatedData(self.axSourceEstimate,self.p,self.estimateBarSuccess,self.estimateBarFail)
        self.fig.canvas.mpl_connect('button_press_event',self.updateFig)
        print('click figure to run next iteration')
        plt.show()
        # now just waiting for user to trigger the event interrupt
        
    def updateFig(self,event):
        # update theta/plot
        self.dataIndex = self.dataIndex + 1
        if self.dataIndex < np.shape(self.data)[0]:
            self.theta = self.conjugate.updateTheta(self.data[self.dataIndex],self.theta)  # n, datum, and theta, respectively
            self.axSourceEstimate.clear() # lazy way, but the built-in figure functions have code for adjusting axis-limits, etc, which help. 
            self.axPosterior.clear()
            self.estimateBarSuccess,self.estimateBarFail = self.conjugate.plotAnalysisOutput(self.axSourceEstimate,self.axPosterior,self.theta)
            self.conjugate.plotSimulatedData(self.axSourceEstimate,self.p,self.estimateBarSuccess,self.estimateBarFail)
            
            self.axSourceEstimate.set_xticklabels(('failure','*success*'))
            if self.data[self.dataIndex]==0:
                self.axSourceEstimate.set_xticklabels(('*failure*','success'))

            plt.draw()
            print('click figure to run next iteration')
        else:
            print('no more data')





p = .2
numDataPoints = 500
f = InteractiveBernoulliFig(p,numDataPoints)
