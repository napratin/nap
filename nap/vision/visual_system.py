"""A biologically-inspired model of visual perception."""

import logging
import numpy as np
import cv2
import cv2.cv as cv

#import pyNN.neuron as sim
from lumos.context import Context
from lumos.input import InputDevice, run

from ..neuron import Neuron, Population, Projection, MultivariateUniform
from .photoreceptor import Rod, Cone
from .simplified.retina import SimplifiedProjector
from .simplified.visual_cortex import SalienceNeuron, SelectionNeuron


class VisualSystem(object):
  """Complete system for processing dynamic visual input."""
  
  num_rods = 10000  # humans: 90-120 million
  num_cones = 1000  # humans: 4.5-6 million
  num_bipolar_cells = 2000
  num_ganglion_cells = 1000
  num_salience_neurons = 400
  num_selection_neurons = 100
  
  default_image_size = (480, 480)  # (width, height)
  
  def __init__(self, imageSize=default_image_size, timeNow=0.0):
    # * Get context and logger
    self.context = Context.getInstance()
    self.logger = logging.getLogger(__name__)
    
    # * Accept arguments, read parameters (TODO)
    self.imageSize = imageSize  # (width, height)
    self.timeNow = timeNow
    
    # * Structural/spatial members
    self.bounds = np.float32([[0.0, 0.0, 2.0], [self.imageSize[0] - 1, self.imageSize[1] - 1, 4.0]])
    self.center = (self.bounds[0] + self.bounds[1]) / 2
    
    # * Images and related members (TODO do we need to initialize these at all? - new images are generated every update)
    self.imageCenter = (self.imageSize[1] / 2, self.imageSize[0] / 2)
    self.imageShapeC3 = (self.imageSize[1], self.imageSize[0], 3)  # numpy shape for 3 channel images
    self.imageShapeC1 = (self.imageSize[1], self.imageSize[0])  # numpy shape for single channel images
    self.imageTypeInt = np.uint8  # numpy dtype for integer-valued images
    self.imageTypeFloat = np.float32  # numpy dtype for real-valued images
    self.images = dict()
    
    # ** RGB and HSV images
    self.images['BGR'] = np.zeros(self.imageShapeC3, dtype=self.imageTypeInt)
    self.images['HSV'] = np.zeros(self.imageShapeC3, dtype=self.imageTypeInt)
    self.images['H'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeInt)
    self.images['S'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeInt)
    self.images['V'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeInt)
    
    # ** Rod and Cone response images (frequency/hue-dependent)
    self.images['Rod'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.images['Cone'] = dict()  # NOTE dict keys must match names of Cone.cone_types (should this be flattened?)
    self.images['Cone']['S'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.images['Cone']['M'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.images['Cone']['L'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    
    # ** Bipolar cell response images
    self.images['Bipolar'] = dict()
    self.images['Bipolar']['ON'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.images['Bipolar']['OFF'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    
    # ** Ganglion cell response images, the source of cortical feature channels
    # TODO Add more Ganglion cell types with different receptive field properties
    #   'RG' +Red    -Green
    #   'GR' +Green  -Red
    #   'RB' +Red    -Blue
    #   'BR' +Blue   -Red
    #   'BY' +Blue   -Yellow
    #   'YB' +Yellow -Blue
    #   'WK' +White  -Black (currently 'ON')
    #   'KW' +Black  -White (currently 'OFF')
    # NOTE R = L cones, G = M cones, B = S cones
    # NOTE Image shapes (h, w, 1) and (h, w) are not compatible unless we use keepdims=True for numpy operations
    self.images['Ganglion'] = dict()
    self.images['Ganglion']['ON'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.images['Ganglion']['OFF'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.images['Ganglion']['RG'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.images['Ganglion']['GR'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.images['Ganglion']['RB'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.images['Ganglion']['BR'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.images['Ganglion']['BY'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.images['Ganglion']['YB'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    
    # ** Combined response (salience) image
    self.images['Salience'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    
    # ** Spatial attention map with a central (covert) spotlight (currently unused; TODO move to VisualCortex? also, use np.ogrid?)
    #self.image['Attention'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    #cv2.circle(self.image['Attention'], self.imageCenter, self.imageSize[0] / 3, 1.0, cv.CV_FILLED)
    #self.image['Attention'] = cv2.blur(self.image['Attention'], (self.imageSize[0] / 4, self.imageSize[0] / 4))  # coarse blur
    
    # ** Output image(s)
    if self.context.options.gui:
      #self.imageOut = np.zeros(self.imageShapeC3, dtype=self.imageTypeInt)
      # TODO Salience and selection output will be for each feature pathway
      self.imageSalienceOut = np.zeros(self.imageShapeC1, dtype=self.imageTypeInt)  # salience neuron outputs
      self.imageSelectionOut = np.zeros(self.imageShapeC1, dtype=self.imageTypeInt)  # selection neuron outputs
    
    # * Image processing elements
    self.bipolarBlurSize = (5, 5)  # size of blurring kernel used when computing Bipolar cell response
    self.ganglionCenterSurroundKernel = self.imageTypeFloat(
      [ [ -1, -1, -1, -1, -1, -1, -1 ],
        [ -1, -1, -1, -1, -1, -1, -1 ],
        [ -1, -1,  7,  7,  7, -1, -1 ],
        [ -1, -1,  7,  9,  7, -1, -1 ],
        [ -1, -1,  7,  7,  7, -1, -1 ],
        [ -1, -1, -1, -1, -1, -1, -1 ],
        [ -1, -1, -1, -1, -1, -1, -1 ] ])
    self.ganglionCenterSurroundKernel /= np.sum(self.ganglionCenterSurroundKernel)  # normalize
    #self.logger.info("Ganglion center-surround kernel:\n{}".format(self.ganglionCenterSurroundKernel))  # [debug]
    self.ganglionKernelLevels = 4
    self.ganglionKernels = [None] * self.ganglionKernelLevels
    self.ganglionKernels[0] = self.ganglionCenterSurroundKernel
    for i in xrange(1, self.ganglionKernelLevels):
      self.ganglionKernels[i] = cv2.resize(self.ganglionKernels[i - 1], dsize=None, fx=2, fy=2)
      self.ganglionKernels[i] /= np.sum(self.ganglionKernels[i])  # normalize
    #self.logger.info("Ganglion center-surround kernel sizes ({} levels): {}".format(self.ganglionKernelLevels, ", ".join("{}".format(k.shape) for k in self.ganglionKernels)))  # [debug]
    
    # * Neuron Populations and Projections connecting them
    self.populations = dict()  # dict with key = population label
    self.projections = dict()  # mapping from (pre_label, post_label) => projection object
    
    # ** Retinal layers (TODO move this to a separate Retina class?)
    self.createRetina()
    
    # ** Layers in the Visual Cortex (TODO move this to a separate VisualCortex class?)
    self.createVisualCortex()
  
  def update(self, timeNow):
    self.timeNow = timeNow
    self.logger.debug("VisualSystem update @ {}".format(self.timeNow))
    
    # * Get HSV
    self.images['HSV'] = cv2.cvtColor(self.images['BGR'], cv2.COLOR_BGR2HSV)
    self.images['H'], self.images['S'], self.images['V'] = cv2.split(self.images['HSV'])
    
    # * Compute Rod and Cone responses
    # TODO Need non-linear response to hue, sat, val (less dependent on sat, val for cones)
    self.imageRod = self.imageTypeFloat(180 - cv2.absdiff(self.images['H'], Rod.rod_type.hue) % 180) * 255 * self.images['V'] * Rod.rod_type.responseFactor  # hack: use constant sat = 200 to make response independent of saturation
    self.images['Cone']['S'] = self.imageTypeFloat(180 - cv2.absdiff(self.images['H'], Cone.cone_types[0].hue) % 180) * self.images['S'] * self.images['V'] * Cone.cone_types[0].responseFactor
    self.images['Cone']['M'] = self.imageTypeFloat(180 - cv2.absdiff(self.images['H'], Cone.cone_types[1].hue) % 180) * self.images['S'] * self.images['V'] * Cone.cone_types[1].responseFactor
    self.images['Cone']['L'] = self.imageTypeFloat(180 - cv2.absdiff(self.images['H'], Cone.cone_types[2].hue) % 180) * self.images['S'] * self.images['V'] * Cone.cone_types[2].responseFactor
    
    # * Compute Bipolar and Ganglion cell responses
    # ** Blurring is a step that is effectively achieved in biology by horizontal cells
    imageRodBlurred = cv2.blur(self.imageRod, self.bipolarBlurSize)
    self.images['Bipolar']['ON'] = np.clip(self.imageRod - 0.75 * imageRodBlurred, 0.0, 1.0)
    self.images['Bipolar']['OFF'] = np.clip((1.0 - self.imageRod) - 0.75 * (1.0 - imageRodBlurred), 0.0, 1.0)  # same as (1 - ON response)?
    #imagesConeSBlurred = cv2.blur(self.images['Cone']['S'], self.bipolarBlurSize)
    #imagesConeMBlurred = cv2.blur(self.images['Cone']['M'], self.bipolarBlurSize)
    #imagesConeLBlurred = cv2.blur(self.images['Cone']['L'], self.bipolarBlurSize)
    
    # ** Ganglion cells simply add up responses from a (bunch of) central bipolar cell(s) (ON/OFF) and surrounding antagonistic bipolar cells (OFF/ON)
    
    # *** Method 1: Center - Surround
    #imageGanglionCenterON = cv2.filter2D(self.images['Bipolar']['ON'], -1, self.ganglionCenterKernel)
    #imageGanglionSurroundOFF = cv2.filter2D(self.images['Bipolar']['OFF'], -1, self.ganglionSurroundKernel)
    #self.images['Ganglion']['ON'] = 0.75 * imageGanglionCenterON + 0.25 * imageGanglionSurroundOFF
    
    # *** Method 2: Center-Surround kernel
    #self.images['Ganglion']['ON'] = np.clip(cv2.filter2D(self.images['Bipolar']['ON'], -1, self.ganglionCenterSurroundKernel), 0.0, 1.0)
    #self.images['Ganglion']['OFF'] = np.clip(cv2.filter2D(self.images['Bipolar']['OFF'], -1, self.ganglionCenterSurroundKernel), 0.0, 1.0)
    
    # *** Method 3: Multi-level Center-Surround kernels, taking maximum
    self.images['Ganglion']['ON'].fill(0.0)
    self.images['Ganglion']['OFF'].fill(0.0)
    self.images['Ganglion']['RG'].fill(0.0)
    self.images['Ganglion']['GR'].fill(0.0)
    self.images['Ganglion']['RB'].fill(0.0)
    self.images['Ganglion']['BR'].fill(0.0)
    self.images['Ganglion']['BY'].fill(0.0)
    self.images['Ganglion']['YB'].fill(0.0)
    for k in self.ganglionKernels:
      # Rod pathway
      self.images['Ganglion']['ON'] = np.maximum(self.images['Ganglion']['ON'], np.clip(cv2.filter2D(self.images['Bipolar']['ON'], -1, k), 0.0, 1.0))
      self.images['Ganglion']['OFF'] = np.maximum(self.images['Ganglion']['OFF'], np.clip(cv2.filter2D(self.images['Bipolar']['OFF'], -1, k), 0.0, 1.0))
      # Cone pathway
      imageRG = self.images['Cone']['L'] - self.images['Cone']['M']
      imageRB = self.images['Cone']['L'] - self.images['Cone']['S']
      imageBY = self.images['Cone']['S'] - (self.images['Cone']['L'] + self.images['Cone']['M']) / 2
      self.images['Ganglion']['RG'] = np.maximum(self.images['Ganglion']['RG'], np.clip(cv2.filter2D(imageRG, -1, k), 0.0, 1.0))
      self.images['Ganglion']['GR'] = np.maximum(self.images['Ganglion']['GR'], np.clip(cv2.filter2D(-imageRG, -1, k), 0.0, 1.0))
      self.images['Ganglion']['RB'] = np.maximum(self.images['Ganglion']['RB'], np.clip(cv2.filter2D(imageRB, -1, k), 0.0, 1.0))
      self.images['Ganglion']['BR'] = np.maximum(self.images['Ganglion']['BR'], np.clip(cv2.filter2D(-imageRB, -1, k), 0.0, 1.0))
      self.images['Ganglion']['BY'] = np.maximum(self.images['Ganglion']['BY'], np.clip(cv2.filter2D(imageBY, -1, k), 0.0, 1.0))
      self.images['Ganglion']['YB'] = np.maximum(self.images['Ganglion']['YB'], np.clip(cv2.filter2D(-imageBY, -1, k), 0.0, 1.0))
    
    # * Compute combined (salience) image; TODO incorporate attention weighting (spatial, as well as by visual feature)
    # ** Method 1: Max of all Ganglion cell images
    self.images['Salience'].fill(0.0)
    for ganglionType, ganglionImage in self.images['Ganglion'].iteritems():
      self.images['Salience'] = np.maximum(self.images['Salience'], ganglionImage)
    
    #self.images['Salience'] *= self.image['Attention']  # TODO evaluate if this is necessary
    
    # * TODO Compute feature vector of attended region
    
    # * Show output images if in GUI mode
    if self.context.options.gui:
      #cv2.imshow("Hue", self.images['H'])
      #cv2.imshow("Saturation", self.images['S'])
      #cv2.imshow("Value", self.images['V'])
      cv2.imshow("Rod response", self.images['Rod'])
      for coneType, coneImage in self.images['Cone'].iteritems():
        cv2.imshow("{} Cones".format(coneType), coneImage)
      for bipolarType, bipolarImage in self.images['Bipolar'].iteritems():
        cv2.imshow("{} Bipolar cells".format(bipolarType), bipolarImage)
      for ganglionType, ganglionImage in self.images['Ganglion'].iteritems():
        cv2.imshow("{} Ganglion cells".format(ganglionType), ganglionImage)
      cv2.imshow("Salience", self.images['Salience'])
      
      # Designate a representative output image
      self.imageOut = self.images['Salience']
      #_, self.imageOut = cv2.threshold(self.imageOut, 0.15, 1.0, cv2.THRESH_TOZERO)  # apply threshold to remove low-response regions
  
  def createRetina(self):
    # TODO * Create Photoreceptor layer
    # TODO * Create BipolarCell layer
    # TODO * Create GanglionCell layer
    pass
  
  def createVisualCortex(self):
    # * TODO Create several feature pathways, each with a salience and selection layer
    # ** Salience neurons (TODO introduce magno and parvo types)
    self.salienceLayerBounds = np.float32([[0.0, 0.0, 0.0], [self.imageSize[0] - 1, self.imageSize[1] - 1, 0.0]])
    #self.salienceNeuronDistribution = MultivariateNormal(mu=self.center, cov=(np.float32([self.center[0] ** 2, self.center[0] ** 2, 1.0]) * np.identity(3, dtype=np.float32)))
    self.salienceNeuronDistribution = MultivariateUniform(lows=[0.0, 0.0, 0.0], highs=[self.imageSize[1], self.imageSize[0], 0.0])
    self.salienceNeurons = Population(numNeurons=self.num_salience_neurons, timeNow=self.timeNow, neuronTypes=[SalienceNeuron], bounds=self.salienceLayerBounds, distribution=self.salienceNeuronDistribution, retina=self)
    self.salienceNeuronPlotColor = 'coral'
    
    # ** Selection neurons
    self.selectionLayerBounds = np.float32([[0.0, 0.0, 50.0], [self.imageSize[0] - 1, self.imageSize[1] - 1, 50.0]])
    self.selectionNeuronDistribution = MultivariateUniform(lows=[0.0, 0.0, 50.0], highs=[self.imageSize[1], self.imageSize[0], 50.0])
    self.selectionNeurons = Population(numNeurons=self.num_selection_neurons, timeNow=self.timeNow, neuronTypes=[SelectionNeuron], bounds=self.selectionLayerBounds, distribution=self.selectionNeuronDistribution, retina=self)
    self.selectionNeuronPlotColor = 'olive'
    
    # * Connect neuron layers
    # ** Salience neurons to selection neurons
    self.salienceNeurons.connectWith(self.selectionNeurons, maxConnectionsPerNeuron=5)
    
    # ** Selection neurons to themselves (lateral inhibition; TODO make this a re-entrant inhibitory Projection with allow_self_connections=False?)
    for source in self.selectionNeurons.neurons:
      for target in self.selectionNeurons.neurons:
        if source == target: continue
        source.gateNeuron(target)
    
    # * Show neuron layers and connections [debug]
    #plotPopulations([self.salienceNeurons, self.selectionNeurons], populationColors=[self.salienceNeuronPlotColor, self.selectionNeuronPlotColor], showConnections=True, equalScaleZ=True)  # [debug]
    
    # * Top-level interface (TODO replicate for each feature pathway; add neuron response/spike frequency as measure of strength)
    self.selectedNeuron = None  # the last selected SelectionNeuron, mainly for display and top-level output
    self.selectedTime = 0.0  # corresponding timestamp
  
  def createPopulation(self, *args, **kwargs):
    """Create a basic Population with given arguments."""
    return self.addPopulation(Population(*args, **kwargs))
  
  def addPopulation(self, population):
    """Add a given Population to this VisualSystem."""
    #assert isinstance(population, Population)  # allow other Population-like objects?
    assert population.label not in self.populations  # refuse to overwrite existing population with same label
    self.populations.append(population)
    return population
  
  def createProjection(self, presynaptic_population, postsynaptic_population, **kwargs):
    """Create a basic Projection from presynaptic to postsynaptic population, with given keyword arguments."""
    assert presynaptic_population in self.populations and postsynaptic_population in self.populations
    return self.addProjection(Projection(presynaptic_population, postsynaptic_population, **kwargs))
  
  def addProjection(self, projection):
    self.projections.append(projection)
    return projection


class VisionManager(SimplifiedProjector):
  """A version of Projector that uses a simplified Retina."""
  
  def __init__(self, retina=None):
    SimplifiedProjector.__init__(self, retina if retina is not None else VisualSystem())


def test_VisualSystem():
  # Test the visual system, as generated
  Context.createInstance()
  run(VisionManager, description="Test application that uses a SimplifiedProjector to run image input through a VisualSystem instance.")


if __name__ == "__main__":
  test_VisualSystem()
