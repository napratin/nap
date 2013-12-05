"""Basic retina model."""

import logging
from math import sqrt
import random
import itertools
from unittest import TestCase
import numpy as np
import cv2

from lumos.context import Context
from lumos.base import FrameProcessor
from lumos.input import InputDevice, run

from matplotlib.pyplot import figure, show
from matplotlib.colors import hsv_to_rgb

from ..neuron import Neuron, NeuronGroup, MultivariateNormal, SymmetricLogNormal, plotNeuronGroups
from .photoreceptor import Rod, Cone

class Retina:
  """A multi-layered surface for hosting different types of neurons that make up a retina."""
  
  num_rods = 10000  # humans: 90-120 million
  num_cones = 1000  # humans: 4.5-6 million
  
  default_image_size = (480, 480)
  
  def __init__(self, imageSize=default_image_size, timeNow=0.0):
    # Initialize members, parameters
    self.context = Context.getInstance()
    self.logger = logging.getLogger(__name__)
    self.imageSize = imageSize
    self.timeNow = timeNow
    self.bounds = np.float32([[0.0, 0.0, 2.0], [self.imageSize[0] - 1, self.imageSize[1] - 1, 4.0]])
    self.center = (self.bounds[0] + self.bounds[1]) / 2
    self.logger.debug("Retina center: {}, image size: {}".format(self.center, self.imageSize))
    self.rodDistribution = SymmetricLogNormal(mu=5.0, sigma=0.5, center=self.center)
    self.rodPlotColor = 'darkmagenta'
    self.coneDistribution = MultivariateNormal(mu=self.center, cov=(np.float32([500.0, 500.0, 1.0]) * np.identity(3, dtype=np.float32)))
    # TODO Create cone populations of different types with their respective spatial distributions (e.g. blue cones are mostly spread out)
    self.conePlotColor = 'darkgreen'
    self.conePlotColorsByType = [hsv_to_rgb(np.float32([[[coneType.hueResponse.mu / 180.0, 1.0, 1.0]]]))[0, 0] for coneType in Cone.cone_types]
    
    # Image and related members
    self.imageBGR = np.zeros((self.imageSize[1], self.imageSize[0], 3), dtype=np.uint8)
    self.imageHSV = np.zeros((self.imageSize[1], self.imageSize[0], 3), dtype=np.uint8)
    if self.context.options.gui:
      self.imageOut = np.zeros((self.imageSize[1], self.imageSize[0], 3), dtype=np.uint8)
    
    # Create neuron groups
    self.rods = NeuronGroup(numNeurons=self.num_rods, timeNow=timeNow, neuronTypes=[Rod], bounds=self.bounds, distribution=self.rodDistribution, retina=self)
    self.cones = NeuronGroup(numNeurons=self.num_cones, timeNow=timeNow, neuronTypes=[Cone], bounds=self.bounds, distribution=self.coneDistribution, retina=self)
    
    # TODO Configure neuron groups (set cone sensitivity, etc.), if needed
  
  def update(self, timeNow):
    self.timeNow = timeNow
    self.logger.debug("Retina update @ {}".format(self.timeNow))
    self.imageHSV = cv2.cvtColor(self.imageBGR, cv2.COLOR_BGR2HSV)
    for photoreceptor in itertools.chain(self.rods.neurons, self.cones.neurons):
      photoreceptor.updateWithP(self.timeNow)  # update probabilistically
      if self.context.options.gui:
        self.imageOut[photoreceptor.pixel[1], photoreceptor.pixel[0], :] = photoreceptor.pixelValue  # render
  
  def plotPhotoreceptors3D(self):
    plotNeuronGroups([self.rods, self.cones], groupColors=[self.rodPlotColor, self.conePlotColor], showConnections=False, equalScaleZ=True)
  
  def plotPhotoreceptorDensities(self, ax=None):
    # Check if axis has been supplied; if not, create new single-axis (-plot) figure
    standalone = False
    if ax is None:
      standalone = True
      fig = figure()
      ax = fig.gca()  # effectively same as fig.add_subplot(111)
    
    # Histogram parameters to bin X values over a thin strip
    numBins = 100
    stripLimitsY = (self.center[1] - self.imageSize[1] / 20.0, self.center[1] + self.imageSize[1] / 20.0)
    
    # Compute cell parameters - each cell is a unit area on which photoreceptors are counted
    cellWidth = (self.bounds[1, 0] - self.bounds[0, 0]) / numBins
    cellHeight = stripLimitsY[1] - stripLimitsY[0]
    cellArea = cellWidth * cellHeight
    
    # Pick photoreceptor locations that lie within strip
    rodLocsInStrip = self.rods.neuronLocations[(stripLimitsY[0] <= self.rods.neuronLocations[:, 1]) & (self.rods.neuronLocations[:, 1] <= stripLimitsY[1])]
    coneLocsInStrip = self.cones.neuronLocations[(stripLimitsY[0] <= self.cones.neuronLocations[:, 1]) & (self.cones.neuronLocations[:, 1] <= stripLimitsY[1])]
    
    # Plot histogram of photoreceptor densities
    n, bins, patches = ax.hist(rodLocsInStrip[:, 0].T, bins=numBins, range=(self.bounds[0, 0], self.bounds[1, 0]), color=self.rodPlotColor, alpha=0.5, histtype='stepfilled', label='Rods')  # plot rod histogram
    ax.hist(coneLocsInStrip[:, 0].T, bins=bins, range=(self.bounds[0, 0], self.bounds[1, 0]), color=self.conePlotColor, alpha=0.5, histtype='stepfilled', label='Cones')  # plot cone histogram, using the same bins as for rods
    #n, bins, patches = hist([rodLocsInStrip[:, 0].T, coneLocsInStrip[:, 0].T], bins=100, range=(self.bounds[0, 0], self.bounds[1, 0]), color=[self.rodPlotColor, self.conePlotColor], alpha=0.5, histtype='stepfilled', label=['Rods', 'Cones'])  # combined
    ax.set_xlabel("Position (pixels)")
    ax.set_ylabel("Density (# per {:.2f}*{:.2f} pixel^2 area)".format(cellWidth, cellHeight))
    ax.set_title("Photoreceptor density in simulated retina")
    ax.legend()
    
    if standalone:
      show()
  
  def plotConeSensitivities(self, ax=None):
    # Check if axis has been supplied; if not, create new single-axis (-plot) figure
    standalone = False
    if ax is None:
      standalone = True
      fig = figure()
      ax = fig.gca()  # effectively same as fig.add_subplot(111)
    
    # Histogram parameters to bin sensitivity over hue range
    numBins = 60
    
    '''
    # Plot histogram of cone sensitivities (hues)
    #coneHues = [cone.hue for cone in self.cones.neurons]  # all hues, no grouping
    coneHuesByType = [[cone.hue for cone in coneSet] for coneSet in ((cone for cone in self.cones.neurons if cone.coneType == coneType) for coneType in Cone.cone_types)]  # hues grouped by type
    nums, bins, patches = ax.hist(coneHuesByType, bins=numBins, range=(0, 180), color=self.conePlotColorsByType, alpha=0.8, histtype='stepfilled', label='Cone types')
    ax.set_xlabel("Hue (degrees, 0..180)")
    ax.set_ylabel("Count (# of cones)")
    ax.set_title("Cone sensitivity distribution in simulated retina")
    ax.legend([coneType.name for coneType in Cone.cone_types])
    
    # Plot histogram of cone sensitivities (frequencies)
    #coneFreqs = [cone.freq for cone in self.cones.neurons]  # all frequencies, no grouping
    coneFreqsByType = [[cone.freq for cone in coneSet] for coneSet in ((cone for cone in self.cones.neurons if cone.coneType == coneType) for coneType in Cone.cone_types)]  # frequencies grouped by type
    nums, bins, patches = ax.hist(coneFreqsByType, bins=numBins, color=self.conePlotColorsByType, alpha=0.8, histtype='stepfilled', label='Cone types')
    ax.set_ylim([0, np.max(nums) + 1])  # NOTE this shouldn't be needed, but without it Y-axis is not getting scaled properly
    ax.set_xlabel("Frequency (nm)")
    ax.set_ylabel("Count (# of cones)")
    ax.set_title("Cone sensitivity distribution in simulated retina")
    ax.legend([coneType.name for coneType in Cone.cone_types])
    '''
    # Plot histogram of cone sensitivities (frequency responses)
    #coneFreqs = [cone.freq for cone in self.cones.neurons]  # all frequencies, no grouping
    #coneSens = [cone.coneType.sensitivity for cone in self.cones.neurons]  # all sensitivities, no grouping
    coneFreqsByType = [[cone.freq for cone in coneSet] for coneSet in ((cone for cone in self.cones.neurons if cone.coneType == coneType) for coneType in Cone.cone_types)]  # frequencies grouped by type
    coneSensByType = [[cone.coneType.sensitivity for cone in coneSet] for coneSet in ((cone for cone in self.cones.neurons if cone.coneType == coneType) for coneType in Cone.cone_types)]  # sensitivities grouped by type
    nums, bins, patches = ax.hist(coneFreqsByType, weights=coneSensByType, bins=numBins, color=self.conePlotColorsByType, alpha=0.8, histtype='stepfilled', label='Cone types')
    ax.set_ylim([0, np.max(nums)])  # NOTE this shouldn't be needed, but without it Y-axis is not getting scaled properly
    ax.set_xlabel("Frequency (nm)")
    ax.set_ylabel("Weighted count (# of cones * sensitivity)")
    ax.set_title("Cone response distribution in simulated retina")
    ax.legend([coneType.name for coneType in Cone.cone_types])
    
    #print "\n".join("{}: {} {} {}".format(bin, n0, n1, n2) for bin, n0, n1, n2 in zip(bins, nums[0], nums[1], nums[2]))  # [debug]
    
    if standalone:
      show()


class Projector(FrameProcessor):
  """An input manager that correctly projects incoming images onto the retina, with a movable point of focus."""
  
  key_focus_jump = 10  # no. of pixels to shift focus under (manual) keyboard control
  
  def __init__(self, retina=None):
    FrameProcessor.__init__(self)
    self.retina = retina if retina is not None else Retina()
  
  def initialize(self, imageIn, timeNow):
    FrameProcessor.initialize(self, imageIn, timeNow)
    self.screenSize = (self.imageSize[0] + 2 * self.retina.imageSize[0], self.imageSize[1] + 2 * self.retina.imageSize[1])  # create a screen which is big enough to accomodate input image and allow panning retina's focus to the edges
    self.logger.debug("Screen size: {}".format(self.screenSize))
    self.screen = np.zeros((self.screenSize[1], self.screenSize[0], 3), dtype=np.uint8)
    self.updateImageRect()
    self.setFocus(self.screenSize[0] / 2, self.screenSize[1] / 2)  # calls updateFocusRect()
  
  def process(self, imageIn, timeNow):
    self.image = imageIn
    # Copy image to screen, and part of screen to retina (TODO optimize this to a single step?)
    self.screen[self.imageRect[2]:self.imageRect[3], self.imageRect[0]:self.imageRect[1]] = self.image
    #if self.context.options.gui: cv2.imshow("Screen", self.screen)  # [debug]
    self.retina.imageBGR[:] = self.screen[self.focusRect[2]:self.focusRect[3], self.focusRect[0]:self.focusRect[1]]
    #if self.context.options.gui: cv2.imshow("Retina", self.retina.imageBGR)  # [debug]
    self.retina.update(timeNow)
    return True, self.retina.imageOut
  
  def onKeyPress(self, key, keyChar=None):
    if keyChar == 'w':
      self.shiftFocus(deltaY=-self.key_focus_jump)
    elif keyChar == 's':
      self.shiftFocus(deltaY=self.key_focus_jump)
    elif keyChar == 'a':
      self.shiftFocus(deltaX=-self.key_focus_jump)
    elif keyChar == 'd':
      self.shiftFocus(deltaX=self.key_focus_jump)
    elif keyChar == 'c':
      self.setFocus(self.screenSize[0] / 2, self.screenSize[1] / 2)
    return True
  
  def updateImageRect(self):
    # Compute image rect bounds - constant screen area where image is copied: (left, right, top, bottom)
    # TODO Ensure rect format (left, right, top, bottom) doesn't clash with OpenCV convention (left, top, width, height)
    #      Or, create a versatile utility class Rect with appropriate properties and conversions
    left = self.screenSize[0] / 2 - self.imageSize[0] / 2
    top = self.screenSize[1] / 2 - self.imageSize[1] / 2
    self.imageRect = np.int_([left, left + self.imageSize[0], top, top + self.imageSize[1]])
    self.logger.debug("Image rect: {}".format(self.imageRect))
  
  def shiftFocus(self, deltaX=0, deltaY=0):
    self.setFocus(self.focusPoint[0] + deltaX, self.focusPoint[1] + deltaY)
  
  def setFocus(self, x, y):
    self.focusPoint = (np.clip(x, self.imageRect[0], self.imageRect[1] - 1), np.clip(y, self.imageRect[2], self.imageRect[3] - 1))
    self.updateFocusRect()
  
  def updateFocusRect(self):
    # Compute focus rect bounds - varying screen area that is copied to retina: (left, right, top, bottom)
    left = self.focusPoint[0] - self.retina.imageSize[0] / 2
    top = self.focusPoint[1] - self.retina.imageSize[1] / 2
    self.focusRect = np.int_([left, left + self.retina.imageSize[0], top, top + self.retina.imageSize[1]])
    self.logger.debug("Focus rect: {}".format(self.focusRect))


class TestRetina(TestCase):
  # TODO Move to nap.vision.tests?
  def setUp(self):
    Context.createInstance()
  
  def test_photoreceptors(self):
    retina = Retina()
    retina.plotPhotoreceptors3D()
    retina.plotPhotoreceptorDensities()
    retina.plotConeSensitivities()
  
  def test_projector(self):
    run(Projector, description="Test application that uses a Projector to run image input through a Retina.")


if __name__ == "__main__":
  TestRetina('test_projector').run()
