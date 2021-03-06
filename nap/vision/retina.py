"""Basic target model."""

import logging
from math import sqrt
import random
import itertools
import argparse
import inspect
from unittest import TestCase
import numpy as np
import cv2
from collections import OrderedDict

from lumos.context import Context
from lumos.input import Projector, run

from matplotlib.pyplot import figure, show, hold, pause
from matplotlib.colors import hsv_to_rgb

from ..util.quadtree import Rect
from ..neuron import Neuron, Population, GrowthCone, MultivariateNormal, SymmetricLogNormal, plotPopulations
from .photoreceptor import Rod, Cone
from .bipolar import BipolarCell


class Retina(object):  # should we ihherit from FrameProcessor?
  """A multi-layered surface for hosting different types of neurons that make up a target.
  
  [Deprecated] Use VisualSystem instead.
  
  """
  
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
    self.imageCenter = (self.imageSize[1] / 2, self.imageSize[0] / 2)
    self.imageShapeC3 = (self.imageSize[1], self.imageSize[0], 3)  # numpy shape for 3 channel images
    self.imageShapeC1 = (self.imageSize[1], self.imageSize[0])  # numpy shape for single channel images
    # NOTE Image shapes (h, w, 1) and (h, w) are not compatible unless we use keepdims=True for numpy operations
    self.imageTypeInt = np.uint8  # numpy dtype for integer-valued images
    self.imageTypeFloat = np.float32  # numpy dtype for real-valued images
    self.images = OrderedDict()
    
    # ** RGB and HSV images
    self.images['BGR'] = np.zeros(self.imageShapeC3, dtype=self.imageTypeInt)
    self.images['HSV'] = np.zeros(self.imageShapeC3, dtype=self.imageTypeInt)
    self.images['H'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeInt)
    self.images['S'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeInt)
    self.images['V'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeInt)
    
    # ** Freq/hue-dependent response images for rods and different cone types
    self.imageRod = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.imagesCone = dict()  # NOTE dict keys must match names of Cone.cone_types
    self.imagesCone['S'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.imagesCone['M'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.imagesCone['L'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    
    # ** Output image(s)
    self.imageOut = None
    if self.context.options.gui:
      self.imageOut = np.zeros(self.imageShapeC3, dtype=self.imageTypeInt)
      self.imagesBipolar = dict()
      self.imagesBipolar['ON'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeInt)
      self.imagesBipolar['OFF'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeInt)
    
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
  
  def initialize(self, imageIn, timeNow):
    pass  # to emulate FrameProcessor-like interface
  
  def process(self, imageIn, timeNow):
    self.timeNow = timeNow
    self.logger.debug("Retina update @ {}".format(self.timeNow))
    self.images['BGR'][:] = imageIn
    self.images['HSV'] = cv2.cvtColor(self.images['BGR'], cv2.COLOR_BGR2HSV)
    self.images['H'], self.images['S'], self.images['V'] = cv2.split(self.images['HSV'])
    # TODO Need non-linear response to hue, sat, val (less dependent on sat, val for cones)
    self.imageRod = np.float32(180 - cv2.absdiff(self.images['H'], Rod.rod_type.hue) % 180) * 255 * self.images['V'] * Rod.rod_type.responseFactor  # hack: use constant sat = 200 to make response independent of saturation
    self.imagesCone['S'] = np.float32(180 - cv2.absdiff(self.images['H'], Cone.cone_types[0].hue) % 180) * self.images['S'] * self.images['V'] * Cone.cone_types[0].responseFactor
    self.imagesCone['M'] = np.float32(180 - cv2.absdiff(self.images['H'], Cone.cone_types[1].hue) % 180) * self.images['S'] * self.images['V'] * Cone.cone_types[1].responseFactor
    self.imagesCone['L'] = np.float32(180 - cv2.absdiff(self.images['H'], Cone.cone_types[2].hue) % 180) * self.images['S'] * self.images['V'] * Cone.cone_types[2].responseFactor
    if self.context.options.gui:
      #cv2.imshow("Hue", self.images['H'])
      #cv2.imshow("Saturation", self.images['S'])
      #cv2.imshow("Value", self.images['V'])
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
    
    return True, self.imageOut
  
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
    ax.set_title("Photoreceptor density in simulated target")
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
    ax.set_title("Cone sensitivity distribution in simulated target")
    ax.legend([coneType.name for coneType in Cone.cone_types])
    
    # Plot histogram of cone sensitivities (frequencies)
    #coneFreqs = [cone.freq for cone in self.cones.neurons]  # all frequencies, no grouping
    coneFreqsByType = [[cone.freq for cone in coneSet] for coneSet in ((cone for cone in self.cones.neurons if cone.coneType == coneType) for coneType in Cone.cone_types)]  # frequencies grouped by type
    nums, bins, patches = ax.hist(coneFreqsByType, bins=numBins, color=self.conePlotColorsByType, alpha=0.8, histtype='stepfilled', label='Cone types')
    ax.set_ylim([0, np.max(nums) + 1])  # NOTE this shouldn't be needed, but without it Y-axis is not getting scaled properly
    ax.set_xlabel("Frequency (nm)")
    ax.set_ylabel("Count (# of cones)")
    ax.set_title("Cone sensitivity distribution in simulated target")
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
    run(Projector(Retina()), description="Test application that uses a Projector to run image input through a Retina.")
  
  def test_rod_potential(self):
    from ..neuron import action_potential_trough, action_potential_peak
    
    class MonitoringProjector(Projector):
      do_plot = False
      def __init__(self, retina=None):
        Projector.__init__(self, retina)
        
        # Neuron to monitor
        self.testRodIdx = 0
        self.testRod = self.target.rods.neurons[self.testRodIdx]
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
          self.testRodIdx = (self.testRodIdx + 1) % len(self.target.rods.neurons)
          self.testRod = self.target.rods.neurons[self.testRodIdx]
          self.logger.info("[>] Test rod [{}]: {}".format(self.testRodIdx, self.testRod))
        elif keyChar == ',':
          self.testRodIdx = (self.testRodIdx - 1) % len(self.target.rods.neurons)
          self.testRod = self.target.rods.neurons[self.testRodIdx]
          self.logger.info("[<] Test rod [{}]: {}".format(self.testRodIdx, self.testRod))
        else:
          return Projector.onKeyPress(self, key, keyChar)
        return True
    
    print "Running MonitoringProjector instance..."
    run(MonitoringProjector(Projector), description="Retina processing with monitor on a single neuron.")


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
