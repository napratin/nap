"""Basic retina model."""

import logging
from math import sqrt
import numpy as np
import cv2

from matplotlib.pyplot import figure, plot, hist, legend, xlabel, ylabel, title, show
from scipy.stats.kde import gaussian_kde

from ..neuron import Neuron, NeuronGroup, MultivariateNormal, SymmetricLogNormal, plotNeuronGroups
from .photoreceptor import Rod, Cone

class Retina:
  """A multi-layered surface for hosting different types of neurons that make up a retina."""
  
  num_rods = 1000  # humans: 90-120 million
  num_cones = 50  # humans: 4.5-6 million
  
  def __init__(self, timeNow=0.0):
    self.timeNow = timeNow
    
    self.logger = logging.getLogger(__name__)
    self.bounds = np.float32([[0.0, 0.0, 2.0], [640.0, 480.0, 4.0]])
    self.center = (self.bounds[0] + self.bounds[1]) / 2
    self.size = self.bounds[1] - self.bounds[0]
    self.logger.debug("Retina center: {}, size: {}".format(self.center, self.size))
    self.rodDistribution = SymmetricLogNormal(mu=5.0, sigma=0.5, center=self.center)
    self.coneDistribution = MultivariateNormal(mu=self.center, cov=(np.float32([500.0, 500.0, 1.0]) * np.identity(3, dtype=np.float32)))
    
    self.rods = NeuronGroup(numNeurons=self.num_rods, timeNow=timeNow, neuronTypes=[Rod], bounds=self.bounds, distribution=self.rodDistribution)
    self.cones = NeuronGroup(numNeurons=self.num_cones, timeNow=timeNow, neuronTypes=[Cone], bounds=self.bounds, distribution=self.coneDistribution)
  
  def plotPhotoreceptorDensities(self, ax=None):
    # Check if axis has been supplied; if not, create new single-axis (-plot) figure
    standalone = False
    if ax is None:
      standalone = True
      fig = figure()
      ax = fig.gca()  # effectively same as fig.add_subplot(111)
    
    # Histogram parameters to bin X values over a thin strip
    numBins = 100
    stripLimitsY = (self.center[1] - self.size[1] / 20, self.center[1] + self.size[1] / 20)
    
    # Compute cell parameters - each cell is a unit area on which photoreceptors are counted
    cellWidth = (self.bounds[1, 0] - self.bounds[0, 0]) / numBins
    cellHeight = stripLimitsY[1] - stripLimitsY[0]
    cellArea = cellWidth * cellHeight
    
    # Pick photoreceptor locations that lie within strip
    rodLocsInStrip = self.rods.neuronLocations[(stripLimitsY[0] <= self.rods.neuronLocations[:, 1]) & (self.rods.neuronLocations[:, 1] <= stripLimitsY[1])]
    coneLocsInStrip = self.cones.neuronLocations[(stripLimitsY[0] <= self.cones.neuronLocations[:, 1]) & (self.cones.neuronLocations[:, 1] <= stripLimitsY[1])]
    
    # Plot histogram of photoreceptor densities
    n, bins, patches = ax.hist(rodLocsInStrip[:, 0].T, bins=numBins, range=(self.bounds[0, 0], self.bounds[1, 0]), color='darkmagenta', alpha=0.5, histtype='stepfilled', label='Rods')  # plot rod histogram
    ax.hist(coneLocsInStrip[:, 0].T, bins=bins, range=(self.bounds[0, 0], self.bounds[1, 0]), color='darkgreen', alpha=0.5, histtype='stepfilled', label='Cones')  # plot cone histogram, using the same bins as for rods
    #n, bins, patches = hist([rodLocsInStrip[:, 0].T, coneLocsInStrip[:, 0].T], bins=100, range=(self.bounds[0, 0], self.bounds[1, 0]), color=['darkmagenta', 'darkgreen'], alpha=0.5, histtype='stepfilled', label=['Rods', 'Cones'])  # combined
    ax.set_xlabel("Position (pixels)")
    ax.set_ylabel("Density (# per {:.2f}*{:.2f} pixel^2 area)".format(cellWidth, cellHeight))
    ax.set_title("Photoreceptor density in simulated self")
    ax.legend()
    
    if standalone:
      show()


def test_photoreceptors():
  logging.basicConfig(format="%(levelname)s | %(name)s | %(funcName)s() | %(message)s", level=logging.DEBUG)  # sets up basic logging, if it's not already configured
  timeNow = 0.0
  retina = Retina(timeNow)
  plotNeuronGroups([retina.rods, retina.cones], groupColors=['darkmagenta', 'darkgreen'], showConnections=False, equalScaleZ=True)
  retina.plotPhotoreceptorDensities()


if __name__ == "__main__":
  test_photoreceptors()
