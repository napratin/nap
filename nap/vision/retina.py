"""Basic retina model."""

import logging
from math import sqrt
import random
import itertools
import argparse
import inspect
from unittest import TestCase
import numpy as np
import cv2

from lumos.context import Context
from lumos.base import FrameProcessor
from lumos.input import InputDevice, run

from matplotlib.pyplot import figure, show, hold, pause
from matplotlib.colors import hsv_to_rgb

from ..util.quadtree import Rect
from ..neuron import Neuron, Population, GrowthCone, MultivariateNormal, SymmetricLogNormal, plotPopulations
from .photoreceptor import Rod, Cone
from .bipolar import BipolarCell

class Retina:
  """A multi-layered surface for hosting different types of neurons that make up a retina."""
  
  num_rods = 10000  # humans: 90-120 million
  num_cones = 1000  # humans: 4.5-6 million
  num_bipolar_cells = 2000
  
  default_image_size = (480, 480)
  
  def __init__(self, imageSize=default_image_size, timeNow=0.0):
    # * Initialize members, parameters
    self.context = Context.getInstance()
    self.logger = logging.getLogger(__name__)
    self.imageSize = imageSize
    self.timeNow = timeNow
    self.bounds = np.float32([[0.0, 0.0, 2.0], [self.imageSize[0] - 1, self.imageSize[1] - 1, 4.0]])
    self.center = (self.bounds[0] + self.bounds[1]) / 2
    self.logger.debug("Retina center: {}, image size: {}".format(self.center, self.imageSize))
    self.rodDistribution = SymmetricLogNormal(mu=5.0, sigma=0.5, center=self.center)
    self.rodPlotColor = 'darkmagenta'
    self.coneDistribution = MultivariateNormal(mu=self.center, cov=(np.float32([1000.0, 1000.0, 1.0]) * np.identity(3, dtype=np.float32)))
    # TODO Create cone populations of different types with their respective spatial distributions (e.g. blue cones are mostly spread out)
    self.conePlotColor = 'darkgreen'
    self.conePlotColorsByType = [hsv_to_rgb(np.float32([[[coneType.hueResponse.mu / 180.0, 1.0, 1.0]]]))[0, 0] for coneType in Cone.cone_types]
    self.bipolarCellDistribution = MultivariateNormal(mu=self.center + np.float32([0.0, 0.0, 10.0]), cov=(np.float32([16000.0, 16000.0, 1.0]) * np.identity(3, dtype=np.float32)))
    self.bipolarCellPlotColor = 'orange'
    
    # * Image and related members
    # ** RGB and HSV images
    self.imageBGR = np.zeros((self.imageSize[1], self.imageSize[0], 3), dtype=np.uint8)
    self.imageHSV = np.zeros((self.imageSize[1], self.imageSize[0], 3), dtype=np.uint8)
    self.imageH = np.zeros((self.imageSize[1], self.imageSize[0], 1), dtype=np.uint8)
    self.imageS = np.zeros((self.imageSize[1], self.imageSize[0], 1), dtype=np.uint8)
    self.imageV = np.zeros((self.imageSize[1], self.imageSize[0], 1), dtype=np.uint8)
    
    # ** Freq/hue-dependent response images for rods and different cone types
    self.imageRod = np.zeros((self.imageSize[1], self.imageSize[0], 1), dtype=np.float32)
    self.imagesCone = dict()  # NOTE dict keys must match names of Cone.cone_types
    self.imagesCone['S'] = np.zeros((self.imageSize[1], self.imageSize[0], 1), dtype=np.float32)
    self.imagesCone['M'] = np.zeros((self.imageSize[1], self.imageSize[0], 1), dtype=np.float32)
    self.imagesCone['L'] = np.zeros((self.imageSize[1], self.imageSize[0], 1), dtype=np.float32)
    
    # ** Output image(s)
    if self.context.options.gui:
      self.imageOut = np.zeros((self.imageSize[1], self.imageSize[0], 3), dtype=np.uint8)
      self.imagesBipolar = dict()
      self.imagesBipolar['ON'] = np.zeros((self.imageSize[1], self.imageSize[0], 1), dtype=np.uint8)
      self.imagesBipolar['OFF'] = np.zeros((self.imageSize[1], self.imageSize[0], 1), dtype=np.uint8)
    
    # * Create neuron populations
    # ** Photoreceptors
    self.rods = Population(numNeurons=self.num_rods, timeNow=self.timeNow, neuronTypes=[Rod], bounds=self.bounds, distribution=self.rodDistribution, retina=self)
    self.cones = Population(numNeurons=self.num_cones, timeNow=self.timeNow, neuronTypes=[Cone], bounds=self.bounds, distribution=self.coneDistribution, retina=self)
    self.coneTypeNames = [coneType.name for coneType in Cone.cone_types]  # mainly for plotting
    
    # ** Bipolar cells
    self.bipolarCells = Population(numNeurons=self.num_bipolar_cells, timeNow=self.timeNow, neuronTypes=[BipolarCell], bounds=self.bounds, distribution=self.bipolarCellDistribution, retina=self)
    
    # * Connect neuron populations
    growthConeDirection = self.bipolarCells.distribution.mu - self.cones.distribution.mu  # NOTE only using cone distribution center
    growthConeDirection /= np.linalg.norm(growthConeDirection, ord=2)  # need a unit vector
    self.cones.connectWith(self.bipolarCells, maxConnectionsPerNeuron=10, growthCone=GrowthCone(growthConeDirection, spreadFactor=1))
    self.rods.connectWith(self.bipolarCells, maxConnectionsPerNeuron=25, growthCone=GrowthCone(growthConeDirection, spreadFactor=1))
    # TODO Connection currently takes a long time; speed this up with better parameterization and spatial search
  
  def update(self, timeNow):
    self.timeNow = timeNow
    self.logger.debug("Retina update @ {}".format(self.timeNow))
    self.imageHSV = cv2.cvtColor(self.imageBGR, cv2.COLOR_BGR2HSV)
    self.imageH, self.imageS, self.imageV = cv2.split(self.imageHSV)
    # TODO Need non-linear response to hue, sat, val (less dependent on sat, val for cones)
    self.imageRod = np.float32(180 - cv2.absdiff(self.imageH, Rod.rod_type.hue) % 180) * 255 * self.imageV * Rod.rod_type.responseFactor  # hack: use constant sat = 200 to make response independent of saturation
    self.imagesCone['S'] = np.float32(180 - cv2.absdiff(self.imageH, Cone.cone_types[0].hue) % 180) * self.imageS * self.imageV * Cone.cone_types[0].responseFactor
    self.imagesCone['M'] = np.float32(180 - cv2.absdiff(self.imageH, Cone.cone_types[1].hue) % 180) * self.imageS * self.imageV * Cone.cone_types[1].responseFactor
    self.imagesCone['L'] = np.float32(180 - cv2.absdiff(self.imageH, Cone.cone_types[2].hue) % 180) * self.imageS * self.imageV * Cone.cone_types[2].responseFactor
    if self.context.options.gui:
      #cv2.imshow("Hue", self.imageH)
      #cv2.imshow("Saturation", self.imageS)
      #cv2.imshow("Value", self.imageV)
      cv2.imshow("Rod response", self.imageRod)
      cv2.imshow("S-cone response", self.imagesCone['S'])
      cv2.imshow("M-cone response", self.imagesCone['M'])
      cv2.imshow("L-cone response", self.imagesCone['L'])
    
    for photoreceptor in itertools.chain(self.rods.neurons, self.cones.neurons):
      photoreceptor.updateWithP(self.timeNow)  # update probabilistically
      if self.context.options.gui:
        self.imageOut[photoreceptor.pixel[1], photoreceptor.pixel[0], :] = photoreceptor.pixelValue  # render
    
    for bipolarCell in self.bipolarCells.neurons:
      bipolarCell.update(self.timeNow)  # update every iteration
      #bipolarCell.updateWithP(self.timeNow)  # update probabilistically
      if self.context.options.gui:
        self.imagesBipolar[bipolarCell.bipolarType.name][bipolarCell.pixel[1], bipolarCell.pixel[0]] = bipolarCell.pixelValue  # render
    
    if self.context.options.gui:
      cv2.imshow("ON Bipolar cells", self.imagesBipolar['ON'])
      cv2.imshow("OFF Bipolar cells", self.imagesBipolar['OFF'])
  
  def plotPhotoreceptors3D(self):
    plotPopulations([self.rods, self.cones, self.bipolarCells], populationColors=[self.rodPlotColor, self.conePlotColor, self.bipolarCellPlotColor], showConnections=True, equalScaleZ=True)
  
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
    conesByType = [[cone for cone in self.cones.neurons if cone.coneType == coneType] for coneType in Cone.cone_types]
    
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
    #coneSens = [cone.coneType.hueSensitivity for cone in self.cones.neurons]  # all sensitivities, no grouping
    coneFreqsByType = [[cone.freq for cone in coneSet] for coneSet in conesByType]  # frequencies grouped by type
    coneSensByType = [[cone.coneType.hueSensitivity for cone in coneSet] for coneSet in conesByType]  # sensitivities grouped by type
    nums, bins, patches = ax.hist(coneFreqsByType, weights=coneSensByType, bins=numBins, color=self.conePlotColorsByType, alpha=0.8, histtype='stepfilled', label=self.coneTypeNames)
    ax.set_ylim([0, np.max(nums)])  # NOTE this shouldn't be needed, but without it Y-axis is not getting scaled properly
    ax.set_xlabel("Frequency (nm)")
    ax.set_ylabel("Weighted count (# of cones * sensitivity)")
    ax.set_title("Cone response distribution in simulated retina")
    ax.legend()
    
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
    
    if self.context.options.gui:
      self.imageOut = self.retina.imageOut
    return True, self.imageOut
  
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
  
  def test_rod_potential(self):
    from ..neuron import action_potential_trough, action_potential_peak
    
    class MonitoringProjector(Projector):
      do_plot = False
      def __init__(self, retina=None):
        Projector.__init__(self, retina)
        
        # Neuron to monitor
        self.testRodIdx = 0
        self.testRod = self.retina.rods.neurons[self.testRodIdx]
        self.logger.info("Test rod [{}]: {}".format(self.testRodIdx, self.testRod))
        
        # Plotting
        if self.do_plot:
          self.logger.info("Plotting is enabled")
          self.fig = figure()
          hold(True)
          self.ax = self.fig.gca()
          self.ax.set_ylim(action_potential_trough.mu - 0.01, action_potential_peak + 0.02)
          self.ax.set_title("Neuron")
          self.ax.set_xlabel("Time (s)")
          self.ax.set_ylabel("Membrane potential (V)")
      
      def process(self, imageIn, timeNow):
        keepRunning, imageOut = Projector.process(self, imageIn, timeNow)
        self.testRod.p = 1.0  # make sure it updated every iteration
        if self.do_plot:
          self.testRod.plot()
          pause(0.01)
        print "{}\t{}\t{}\t{}\t{}\t{}".format(timeNow, self.testRod.response, self.testRod.potential, self.testRod.I_e, self.testRod.expDecayFactor, self.testRod.pixelValue)  # [debug, non-GUI]
        #print "{}\t{}\t{}\t{}".format(timeNow, self.testRod.potential, self.testRod.expDecayFactor, self.testRod.pixelValue)  # [debug, non-GUI, for BipolarCells]
        #cv2.circle(imageOut, (self.testRod.pixel[0], self.testRod.pixel[1]), 3, np.uint8([255, 0, 255]))
        imageOut[self.testRod.pixel[1], self.testRod.pixel[0]] = np.uint8([255, 0, 255])
        return keepRunning, imageOut
      
      def onKeyPress(self, key, keyChar=None):
        if keyChar == '.':
          self.testRodIdx = (self.testRodIdx + 1) % len(self.retina.rods.neurons)
          self.testRod = self.retina.rods.neurons[self.testRodIdx]
          self.logger.info("[>] Test rod [{}]: {}".format(self.testRodIdx, self.testRod))
        elif keyChar == ',':
          self.testRodIdx = (self.testRodIdx - 1) % len(self.retina.rods.neurons)
          self.testRod = self.retina.rods.neurons[self.testRodIdx]
          self.logger.info("[<] Test rod [{}]: {}".format(self.testRodIdx, self.testRod))
        else:
          return Projector.onKeyPress(self, key, keyChar)
        return True
    
    print "Running MonitoringProjector instance..."
    run(MonitoringProjector, description="Retina processing with monitor on a single neuron.")


if __name__ == "__main__":
  argParser = argparse.ArgumentParser(add_help=False)
  argParser.add_argument('--test', default="test_projector", help="test case to run (a test_ method in TestRetina)")
  context = Context.createInstance(parent_argparsers=[argParser])
  try:
    runner = TestRetina(context.options.test).run
    if context.options.debug:
      import pdb
      pdb.runcall(runner)
    else:
      runner()
  except ValueError as e:
    print "Invalid test: {}".format(e)
    print "Pick from: {}".format(", ".join(name for name, method in inspect.getmembers(TestRetina, predicate=inspect.ismethod) if name.startswith("test_")))
