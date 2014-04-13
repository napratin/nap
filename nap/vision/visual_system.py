"""A biologically-inspired model of visual perception."""

from math import exp, hypot
import logging
import numpy as np
import cv2
import cv2.cv as cv
from collections import OrderedDict, deque
from itertools import izip

#import pyNN.neuron as sim
from lumos.context import Context
from lumos.util import Enum
from lumos.input import Projector, run
from lumos import rpc

from ..util.buffer import InputBuffer, OutputBuffer, BidirectionalBuffer, BufferAccessError
from ..neuron import Neuron, Population, Projection, neuron_inhibition_period, Uniform, MultivariateUniform, MultivariateNormal, NeuronMonitor, plotPopulations
from .photoreceptor import Rod, Cone
from .simplified.visual_cortex import SalienceNeuron, SelectionNeuron, FeatureNeuron
from ..motion.ocular import EmulatedOcularMotionSystem


# Global variables
default_feature_weight = 0.9  # default weight for a feature pathway, treated as update probability for its neurons
default_feature_weight_rest = 0.25  # default weight for features other than the ones desired

# Global initialization
np.set_printoptions(precision=4, linewidth=120)  # for printing feature vectors: a few decimal places are fine; try not to break lines, especially in log files


class VisualFeaturePathway(object):
  """A collection of connected neuron populations that together compute a particular visual feature."""
  
  def __init__(self, label, populations, projections, output=None, p=default_feature_weight, timeNow=0.0):
    self.label = label
    self.logger = logging.getLogger("{}-pathway".format(self.label))
    self.populations = populations  # order of populations matters here; this is the order in which they will be updated
    self.projections = projections
    #assert output in self.populations  # usually, output is a population, but it can be something else
    self.output = output
    self.timeNow = timeNow
    
    # * Top-level interface (TODO add neuron response/spike frequency as measure of strength)
    self.active = True  # used to selectively update specific pathways
    self.p = p  # update probability
    self.selectedNeuron = None  # the last selected SelectionNeuron, mainly for display and top-level output
    self.selectedTime = 0.0  # corresponding timestamp
    self.logger.debug("Initialized {}".format(self))
  
  def update(self, timeNow):
    self.timeNow = timeNow
    # feature pathway specific updates may need to be carried out externally
  
  def __str__(self):
    return "{obj.label}-pathway: active: {obj.active}, p: {obj.p}, output: {output}".format(obj=self, output=(self.output.neurons[0].potential if self.output is not None and len(self.output.neurons) > 0 else None))


class Finst(object):
  """Finger of INSTantiation: A percept defined by a location in allocentric space, used for modulating attention."""
  
  max_activation = 1.0
  half_life = 5.0
  min_good_activation = 0.1  # FINSTs with activation less than this could be discarded
  
  default_radius = 32
  
  def __init__(self, location, focusPoint, radius=default_radius, timeCreated=0.0, activationCreated=max_activation):
    self.location = location  # egocentric fixation location at time of creation
    self.focusPoint = focusPoint  # allocentric focus point at time of creation
    self.radius = radius  # an indicator of size
    self.timeCreated = timeCreated  # creation time
    self.activationCreated = activationCreated  # a measure of the strength of the FINST upon creation
    self.update(timeCreated)
  
  def update(self, timeNow):
    deltaTime = timeNow - self.timeCreated
    self.activation = self.activationCreated / (2 ** (deltaTime / self.half_life))
  
  def getAdjustedLocation(self, focusPoint):
    return (self.location[0] + self.focusPoint[0] - focusPoint[0], self.location[1] + self.focusPoint[1] - focusPoint[1])
  
  def __str__(self):
    return "<loc: {self.location}, focus: {self.focusPoint}, act: {self.activation:.3f}>".format(self=self)


class VisualSystem(object):
  """Complete system for processing dynamic visual input."""
  
  State = Enum(('NONE', 'FREE', 'SACCADE', 'FIXATE'))
  intents = ['find', 'hold', 'release', 'reset']  # all supported intents
  
  default_image_size = (256, 256)  # (width, height) TODO read from context options
  
  num_rods = 10000  # human: 90-120 million
  num_cones = 1000  # human: 4.5-6 million
  num_bipolar_cells = 2000
  num_ganglion_cells = 1000
  num_salience_neurons = 400
  num_selection_neurons = 100
  num_feature_neurons = 2  # no. of feature neurons per pathway, more implies finer feature resolution
  
  num_finsts = 5  # no. of visual FINSTs
  finst_decay_enabled = False  # if enabled, FINST activations will be updated and those with low activation will be purged
  finst_inhibition_enabled = True  # if active FINST locations are inhibited
  
  max_free_duration = 2.0  # artificial bound to prevent no results in case of very low salience inputs
  min_saccade_duration = 0.05  # human: 0.02s (20ms)
  #max_saccade_duration = 0.5  # human: 0.2s (200ms); not used as we end saccade period when ocular motion stops
  min_fixation_duration = 0.5  # human: 0.1s (100ms), varies based by activity
  max_fixation_duration = 3.0  # human: 0.5s (500ms), varies considerably by activity, affected by cognitive control
  max_hold_duration = 5.0

  min_good_salience = 0.66  # recommended values: 0.66 (filters out most unwanted regions)
  min_saccade_salience = 0.175  # minimum salience required to make a saccade to (otherwise reset to center)
  
  foveal_radius_ratio = 0.2  # fraction of distance from center to corners of the retina that is considered to be in foveal region
  #default_fovea_size = (int(foveal_radius_ratio * default_image_size[0]), int(foveal_radius_ratio * default_image_size[1]))
  default_fovea_size = (100, 100)  # fixed size; specify None to compute using foveal radius and image size in __init__()
  
  central_radius_ratio = 0.5  # radius to mark central region where visual acuity is modest and then falls off with eccentricity
  
  def __init__(self, imageSize=default_image_size, foveaSize=default_fovea_size, timeNow=0.0, showMonitor=None, ocularMotionSystem=None):
    # * Get context and logger
    self.context = Context.getInstance()
    self.logger = logging.getLogger(self.__class__.__name__)
    
    # * Accept arguments, read parameters (TODO)
    self.imageSize = imageSize  # (width, height)
    self.foveaSize = foveaSize
    self.timeNow = timeNow
    self.ocularMotionSystem = ocularMotionSystem  # for eye movements, if available
    
    # * System state
    self.state = self.State.NONE
    self.lastTransitionTime = self.timeNow
    self.hold = False  # hold gaze at a fixed location?
    
    # * Structural/spatial members
    self.bounds = np.float32([[0.0, 0.0, 2.0], [self.imageSize[0] - 1, self.imageSize[1] - 1, 4.0]])
    self.center = (self.bounds[0] + self.bounds[1]) / 2
    
    # * Images and related members (TODO do we need to initialize these at all? - new images are generated every update)
    self.imageCenter = (self.imageSize[1] / 2, self.imageSize[0] / 2)
    self.fovealRadius = hypot(self.imageCenter[0], self.imageCenter[1]) * self.foveal_radius_ratio
    if self.foveaSize is None:
      self.foveaSize = (int(self.fovealRadius * 2), int(self.fovealRadius * 2))
    self.fovealSlice = np.index_exp[int(self.imageCenter[1] - self.foveaSize[1] / 2):int(self.imageCenter[1] + self.foveaSize[1] / 2), int(self.imageCenter[0] - self.foveaSize[0] / 2):int(self.imageCenter[0] + self.foveaSize[0] / 2)]
    self.fixationSlice = self.fovealSlice
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
    
    # ** Rod and Cone response images (frequency/hue-dependent)
    self.images['Rod'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.images['Cone'] = OrderedDict()  # NOTE dict keys must match names of Cone.cone_types (should this be flattened?)
    self.images['Cone']['S'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.images['Cone']['M'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.images['Cone']['L'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    
    # ** Bipolar cell response images
    # NOTE Rod bipolars are ON-center only; they connect to OFF-center Ganglion cells to initiate the dark pathway
    #      Here, an OFF map is computed from the ON map in order to simplify computation only
    self.images['Bipolar'] = OrderedDict()
    self.images['Bipolar']['ON'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.images['Bipolar']['OFF'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.images['Bipolar']['S'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.images['Bipolar']['M'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.images['Bipolar']['L'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    
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
    self.ganglionTypes = ['ON', 'OFF', 'RG', 'GR', 'RB', 'BR', 'BY', 'YB']
    self.featurePlotColors = {'ON': 'gray', 'OFF': 'black', 'RG': 'red', 'GR': 'green', 'RB': 'tomato', 'BR': 'blue', 'BY': 'magenta', 'YB': 'gold'}
    self.numGanglionTypes = np.int_(len(self.ganglionTypes))  # TODO use a single num-features parameter across the board?
    self.numGanglionTypes_inv = 1.0 / self.imageTypeFloat(self.numGanglionTypes)  # [optimization: frequently used quantity]
    self.images['Ganglion'] = OrderedDict()
    for ganglionType in self.ganglionTypes:
      self.images['Ganglion'][ganglionType] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    
    # ** Combined response (salience) image (and related variables)
    self.images['Salience'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    self.maxSalience = 0.0
    self.maxSalienceLoc = (-1, -1)
    
    # ** Spatial weight map with a central soft spotlight (use np.ogrid?)
    self.images['Weight'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    cv2.circle(self.images['Weight'], self.imageCenter, int(self.imageSize[0] * self.central_radius_ratio), 1.0, cv.CV_FILLED)
    self.images['Weight'] = cv2.blur(self.images['Weight'], (self.imageSize[0] / 4, self.imageSize[0] / 4))  # coarse blur
    
    # * Image processing elements
    self.bipolarBlurSize = (11, 11)  # size of blurring kernel used when computing Bipolar cell response
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
    self.populations = OrderedDict()  # dict with key = population label
    self.projections = OrderedDict()  # mapping from (pre_label, post_label) => projection object
    
    # ** Retinal layers (TODO move this to a separate Retina class?)
    self.createRetina()
    
    # ** Layers in the Visual Cortex (TODO move this to a separate VisualCortex class?)
    self.createVisualCortex()  # creates and populates self.featurePathways
    
    # * Eye movement
    self.saccadeSalience = 0.0  # salience of last location we moved to
    self.saccadeTarget = (0, 0)  # center-relative
    #self.lastSaccadeTime = self.timeNow  # [unused]
    self.fixationLoc = None  # not None when fixated
    
    # * FINSTs for maintaining attended locations
    self.finsts = deque(maxlen=self.num_finsts)
    
    # * Output image and plots
    self.imageOut = None
    if self.context.options.gui:
      #self.imageOut = np.zeros(self.imageShapeC3, dtype=self.imageTypeInt)
      # TODO Salience and selection output will be for each feature pathway
      self.imageSalienceOut = np.zeros(self.imageShapeC1, dtype=self.imageTypeInt)  # salience neuron outputs
      self.imageSelectionOut = np.zeros(self.imageShapeC1, dtype=self.imageTypeInt)  # selection neuron outputs
      
      if showMonitor is None:
        showMonitor = self.context.options.gui and self.context.options.debug
      if showMonitor:
        self.neuronPotentialMonitor = NeuronMonitor(show_legend=False)
        for pathwayLabel, featurePathway in self.featurePathways.iteritems():
          # Monitor single feature neuron
          #self.neuronPotentialMonitor.addChannel(label=pathwayLabel, obj=featurePathway.output.neurons[0], color=self.featurePlotColors[pathwayLabel])  # very hard-coded way to access single output neuron!
          # Monitor all feature neurons
          for idx, outputNeuron in enumerate(featurePathway.output.neurons):
            self.neuronPotentialMonitor.addChannel(label="{}_{}".format(pathwayLabel, idx), obj=outputNeuron, color=self.featurePlotColors[pathwayLabel])
        self.neuronPotentialMonitor.start()
    
    # * Buffers - mainly for communication with high-level (cognitive) architectures, other modules
    # TODO Initialize all buffers with proper values
    self.buffers = OrderedDict()
    self.buffers['state'] = OutputBuffer(self.state)
    self.buffers['intent'] = InputBuffer(self.handleIntent)  # receive intent in a callable method
    self.buffers['location'] = BidirectionalBuffer((0, 0))  # center-relative
    self.buffers['size'] = BidirectionalBuffer((0, 0))
    self.buffers['features'] = BidirectionalBuffer()
    self.buffers['weights'] = InputBuffer()
    self.buffers['salience'] = OutputBuffer(0.0)
    self.buffers['match'] = OutputBuffer(0.0)
    
    # * Once initialized, start in FREE state
    self.transition(self.State.FREE)
  
  def initialize(self, imageIn, timeNow):
    pass  # to emulate FrameProcessor-like interface
  
  def process(self, imageIn, timeNow):
    self.timeNow = timeNow
    self.images['BGR'][:] = imageIn  # NOTE: must be pre-allocated and of the same (compatible) shape as imageIn
    if self.context.options.gui:
      cv2.imshow("Retina", self.images['BGR'])
    
    # * State-based pre-processing
    if self.state == self.State.SACCADE:
      # Check for saccade end
      if self.timeNow > (self.lastTransitionTime + self.min_saccade_duration) and not self.ocularMotionSystem.isMoving:
        self.transition(self.State.FIXATE)  # TODO: transition to an intermediate state to check for successful saccade completion
      else:
        return True, self.imageOut  # saccadic suppression - skip further processing if performing a saccade
    
    # * TODO Read input buffers
    weights = self.buffers['weights'].get_in(clear=True)
    if weights is not None:
      self.updateFeatureWeights(weights)
    
    # * Get HSV
    self.images['HSV'] = cv2.cvtColor(self.images['BGR'], cv2.COLOR_BGR2HSV)
    self.images['H'], self.images['S'], self.images['V'] = cv2.split(self.images['HSV'])
    
    # * Compute Rod and Cone responses
    # TODO: Need non-linear response to hue, sat, val (less dependent on sat, val for cones)
    # NOTE: Somehow, PhotoreceptorType.hue must be a numpy array, even if it is length 1, otherwise we hit a TypeError: <unknown> is not a numpy array!
    self.images['Rod'] = self.imageTypeFloat(180 - cv2.absdiff(self.images['H'], Rod.rod_type.hue) % 180) * 255 * self.images['V'] * Rod.rod_type.responseFactor  # hack: use constant sat = 200 to make response independent of saturation
    self.images['Cone']['S'] = self.imageTypeFloat(180 - cv2.absdiff(self.images['H'], Cone.cone_types[0].hue) % 180) * self.images['S'] * self.images['V'] * Cone.cone_types[0].responseFactor
    self.images['Cone']['M'] = self.imageTypeFloat(180 - cv2.absdiff(self.images['H'], Cone.cone_types[1].hue) % 180) * self.images['S'] * self.images['V'] * Cone.cone_types[1].responseFactor
    self.images['Cone']['L'] = self.imageTypeFloat(180 - cv2.absdiff(self.images['H'], Cone.cone_types[2].hue) % 180) * self.images['S'] * self.images['V'] * Cone.cone_types[2].responseFactor
    
    # * Compute Bipolar and Ganglion cell responses
    # ** Bipolar responses: Rods 
    # NOTE Blurring is a step that is effectively achieved in biology by horizontal cells
    imageRodBlurred = cv2.blur(self.images['Rod'], self.bipolarBlurSize)
    self.images['Bipolar']['ON'] = np.clip(self.images['Rod'] - 0.75 * imageRodBlurred, 0.0, 1.0)
    self.images['Bipolar']['OFF'] = np.clip((1.0 - self.images['Rod']) - 0.95 * (1.0 - imageRodBlurred), 0.0, 1.0)  # same as (1 - ON response)? (nope)
    
    # ** Bipolar responses: Cones
    # TODO Add multiscale Cone Bipolars to prevent unwanted response to diffuse illumination
    imagesConeSBlurred = cv2.blur(self.images['Cone']['S'], self.bipolarBlurSize)
    imagesConeMBlurred = cv2.blur(self.images['Cone']['M'], self.bipolarBlurSize)
    imagesConeLBlurred = cv2.blur(self.images['Cone']['L'], self.bipolarBlurSize)
    self.images['Bipolar']['S'] = np.clip(self.images['Cone']['S'] - 0.75 * imagesConeSBlurred, 0.0, 1.0)
    self.images['Bipolar']['M'] = np.clip(self.images['Cone']['M'] - 0.75 * imagesConeMBlurred, 0.0, 1.0)
    self.images['Bipolar']['L'] = np.clip(self.images['Cone']['L'] - 0.75 * imagesConeLBlurred, 0.0, 1.0)
    
    # ** Ganglion cells simply add up responses from a (bunch of) central bipolar cell(s) (ON/OFF) and surrounding antagonistic bipolar cells (OFF/ON)
    
    # *** Method 1: Center - Surround
    #imageGanglionCenterON = cv2.filter2D(self.images['Bipolar']['ON'], -1, self.ganglionCenterKernel)
    #imageGanglionSurroundOFF = cv2.filter2D(self.images['Bipolar']['OFF'], -1, self.ganglionSurroundKernel)
    #self.images['Ganglion']['ON'] = 0.75 * imageGanglionCenterON + 0.25 * imageGanglionSurroundOFF
    
    # *** Method 2: Center-Surround kernel
    #self.images['Ganglion']['ON'] = np.clip(cv2.filter2D(self.images['Bipolar']['ON'], -1, self.ganglionCenterSurroundKernel), 0.0, 1.0)
    #self.images['Ganglion']['OFF'] = np.clip(cv2.filter2D(self.images['Bipolar']['OFF'], -1, self.ganglionCenterSurroundKernel), 0.0, 1.0)
    
    # *** Method 3: Multi-level Center-Surround kernels, taking maximum
    for ganglionImage in self.images['Ganglion'].itervalues():
      ganglionImage.fill(0.0)  # reset all to zero
    for k in self.ganglionKernels:
      # Rod pathway
      self.images['Ganglion']['ON'] = np.maximum(self.images['Ganglion']['ON'], np.clip(cv2.filter2D(self.images['Bipolar']['ON'], -1, k), 0.0, 1.0))
      self.images['Ganglion']['OFF'] = np.maximum(self.images['Ganglion']['OFF'], np.clip(cv2.filter2D(self.images['Bipolar']['OFF'], -1, k), 0.0, 1.0))
      # Cone pathway
      imageRG = self.images['Bipolar']['L'] - self.images['Bipolar']['M']
      imageRB = self.images['Bipolar']['L'] - self.images['Bipolar']['S']
      imageBY = self.images['Bipolar']['S'] - (self.images['Bipolar']['L'] + self.images['Bipolar']['M']) / 2
      self.images['Ganglion']['RG'] = np.maximum(self.images['Ganglion']['RG'], np.clip(cv2.filter2D(imageRG, -1, k), 0.0, 1.0))
      self.images['Ganglion']['GR'] = np.maximum(self.images['Ganglion']['GR'], np.clip(cv2.filter2D(-imageRG, -1, k) * 1.6, 0.0, 1.0))  # TODO: formalize this fixed relative weighting scheme to counter unequal color representation
      self.images['Ganglion']['RB'] = np.maximum(self.images['Ganglion']['RB'], np.clip(cv2.filter2D(imageRB, -1, k), 0.0, 1.0))
      self.images['Ganglion']['BR'] = np.maximum(self.images['Ganglion']['BR'], np.clip(cv2.filter2D(-imageRB, -1, k), 0.0, 1.0))
      self.images['Ganglion']['BY'] = np.maximum(self.images['Ganglion']['BY'], np.clip(cv2.filter2D(imageBY, -1, k), 0.0, 1.0))
      self.images['Ganglion']['YB'] = np.maximum(self.images['Ganglion']['YB'], np.clip(cv2.filter2D(-imageBY, -1, k) * 1.6, 0.0, 1.0))  # TODO: also here
    
    # * Compute combined (salience) image; TODO incorporate attention weighting (spatial, as well as by visual feature)
    # ** Method 1: Max of all Ganglion cell images
    self.images['Salience'].fill(0.0)
    for ganglionType, ganglionImage in self.images['Ganglion'].iteritems():
      #self.images['Salience'] = np.maximum(self.images['Salience'], ganglionImage)
      #self.logger.debug("[Salience] Combining {}".format(self.featurePathways[ganglionType]))  # [verbose]
      self.images['Salience'] = np.maximum(self.images['Salience'], np.sqrt(self.featurePathways[ganglionType].p) * ganglionImage)  # take maximum, scaled by feature pathway probabilities (for display only)
      #self.images['Salience'] = self.images['Salience'] + (self.numGanglionTypes_inv * np.sqrt(self.featurePathways[ganglionType].p) * ganglionImage)  # take normalized sum (mixes up features), scaled by feature pathway probabilities (for display only)
    
    # * Update FINSTs if decay is enabled (otherwise activation doesn't change, FINSTs are purged when there's no more room)
    if self.finst_decay_enabled:
      for finst in self.finsts:
        finst.update(self.timeNow)
      # Remove stale FINSTs (TODO: use priority queue, don't depend on FINSTs being sorted by activation)
      while self.finsts and self.finsts[0].activation < Finst.min_good_activation:
        self.finsts.popleft()
    
    # * Apply inhibition based on FINSTs
    if self.finst_inhibition_enabled and self.finsts:
      self.logger.debug("Current FINSTs: {}".format(", ".join(str(finst) for finst in self.finsts)))
      for finst in self.finsts:
        self.inhibitMapAtFinst(self.images['Salience'], finst)
    
    self.images['Salience'] = cv2.blur(self.images['Salience'], (3, 3))  # blur slightly to smooth out specs
    self.images['Salience'] *= self.images['Weight']  # effectively reduces salience around the edges (which can sometime give artificially high values due to partial receptive fields)
    _, self.maxSalience, _, self.maxSalienceLoc = cv2.minMaxLoc(self.images['Salience'])  # find out most salient location (from combined salience map)
    self.logger.debug("Max. salience value: {:5.3f} @ {}".format(self.maxSalience, self.maxSalienceLoc))  # [verbose]
    
    # * Compute features along each pathway
    for pathwayLabel, featurePathway in self.featurePathways.iteritems():
      if featurePathway.active:
        # ** Update feature pathway populations (TODO find a more reliable way of grabbing salience and selection neuron populations)
        #featurePathway.update(self.timeNow)  # currently doesn't do anything, update populations explicitly
        salienceNeurons = featurePathway.populations[0]
        selectionNeurons = featurePathway.populations[1]
        featureNeurons = featurePathway.populations[2]
        # *** Salience neurons
        for salienceNeuron in salienceNeurons.neurons:
          #salienceNeuron.update(timeNow)  # update every iteration
          #salienceNeuron.updateWithP(timeNow)  # update using intrinsic probability (adaptive)
          if np.random.uniform() < featurePathway.p:  # update using pathway probability (TODO try to make this adaptive?)
            salienceNeuron.update(timeNow)
          #self.logger.debug("{} Salience neuron potential: {:.3f}, response: {:.3f}, I_e: {}, pixelValue: {}".format(pathwayLabel, salienceNeuron.potential, salienceNeuron.response, salienceNeuron.I_e, salienceNeuron.pixelValue))
          
        # *** Selection neurons (TODO mostly duplicated code, perhaps generalizable?)
        for selectionNeuron in selectionNeurons.neurons:
          #selectionNeuron.update(timeNow)  # update every iteration
          #selectionNeuron.updateWithP(timeNow)  # update using intrinsic probability (adaptive)
          if np.random.uniform() < featurePathway.p:  # update using pathway probability (TODO try to make this adaptive?)
            selectionNeuron.update(timeNow)
          else:
            selectionNeuron.potentialAccumulated = 0.0  # clear any accumulated potential, effectively inhibiting the selection neuron
          #self.logger.debug("{} Selection neuron potential: {:.3f}, pixelValue: {}".format(pathwayLabel, selectionNeuron.potential, selectionNeuron.pixelValue))
        
        # **** Pick one selection neuron, inhibit others
        # TODO Use a top-level feature neuron with graded potential to return activation level
        #numUninhibited = 0  # [debug]
        for selectionNeuron in selectionNeurons.neurons:
          # Render selection neuron's position with response-based pixel value (TODO build receptive field when synapses are made, or later, using a stimulus test phase?)
          #if selectionNeuron.pixelValue > 200: print "[{:.2f}] {}".format(timeNow, selectionNeuron)  # [debug]
          if not selectionNeuron.isInhibited and selectionNeuron.timeLastFired == timeNow:  # only deal with uninhibited neurons that just fired in this iteration
            #numUninhibitedFired += 1  # [debug]
            #cv2.circle(self.imageSelectionOut, (selectionNeuron.pixel[0], selectionNeuron.pixel[1]), self.imageSize[0] / 20, selectionNeuron.pixelValue, cv.CV_FILLED)  # only render the one selected neuron, later
            featurePathway.selectedNeuron = selectionNeuron
            featurePathway.selectedTime = timeNow
            featurePathway.selectedNeuron.inhibit(timeNow, neuron_inhibition_period + 0.75)  # inhibit selected neuron for a bit longer
            break  # first uninhibited SelectionNeuron will be our selected neuron
        #print "# Uninhibited selection neurons that fired: {}".format(numUninhibitedFired)  # [debug]
        
        # *** Feature neuron
        for featureNeuron in featureNeurons.neurons:
          featureNeuron.update(timeNow)  # update every iteration
          #featureNeuron.updateWithP(timeNow)  # update probabilistically
          #self.logger.debug("{} Feature neuron potential: {:.3f}, pixelValue: {}".format(pathwayLabel, featureNeuron.potential, featureNeuron.pixelValue))
        
        # ** Render output images and show them (per feature pathway, better show in debug mode only)
        if self.context.options.gui and self.context.options.debug:
          # *** Salience neurons
          self.imageSalienceOut.fill(0.0)
          for salienceNeuron in salienceNeurons.neurons:
            # Render salience neuron's receptive field with response-based pixel value (TODO cache int radii and pixel as tuple?)
            cv2.circle(self.imageSalienceOut, (salienceNeuron.pixel[0], salienceNeuron.pixel[1]), np.int_(salienceNeuron.rfRadius), 128)
            cv2.circle(self.imageSalienceOut, (salienceNeuron.pixel[0], salienceNeuron.pixel[1]), np.int_(salienceNeuron.rfCenterRadius), salienceNeuron.pixelValue, cv.CV_FILLED)
          
          # *** Selection neurons
          if featurePathway.selectedNeuron is not None and (timeNow - featurePathway.selectedTime) < 3.0:
            #self.imageSelectionOut.fill(0.0)
            cv2.circle(self.imageSalienceOut, (featurePathway.selectedNeuron.pixel[0], featurePathway.selectedNeuron.pixel[1]), featurePathway.selectedNeuron.rfRadius, int(255 * exp(featurePathway.selectedTime - timeNow)), 2)  # draw selected neuron with a shade that fades with time (on salience output image)
            #cv2.circle(self.imageSelectionOut, (featurePathway.selectedNeuron.pixel[0], featurePathway.selectedNeuron.pixel[1]), featurePathway.selectedNeuron.rfRadius, int(255 * exp(featurePathway.selectedTime - timeNow)), cv.CV_FILLED)  # draw selected neuron with a shade that fades with time
          
          cv2.imshow("{} Salience".format(pathwayLabel), self.imageSalienceOut)
          #cv2.imshow("{} Selection".format(pathwayLabel), self.imageSelectionOut)
    
    # * TODO Compute feature vector of attended region
    
    # * Post-processing: Write to output buffers, state-based actions, check for transitions
    self.buffers['salience'].set_out(self.maxSalience)
    self.buffers['location'].set_out(self.toCenterRelative(self.maxSalienceLoc))
    self.updateFeatureVector()  # external buffer reads may need this
    if self.state == self.State.FREE:
      if self.maxSalience >= self.min_good_salience or \
         (self.maxSalience >= self.min_saccade_salience and self.timeNow > (self.lastTransitionTime + self.max_free_duration)): # we have good (or good enough) salience, lets saccade to it
        self.saccadeSalience = self.maxSalience
        self.saccadeTarget = np.int_(self.buffers['location'].get_out())  # ocular motion system requires a 2-element numpy array
        self.performSaccade(self.saccadeTarget)
      elif self.timeNow > (self.lastTransitionTime + self.max_free_duration):  # we've been waiting too long, nothing significant, let's reset
        self.performSaccade(None)  # TODO: Probabilistically choose a not-so-good location?
    elif self.state == self.State.FIXATE:
      # Update fixation location (first time this fixation only)
      # TODO: Maybe a good idea to use a new FIXATED state after FIXATE?
      if self.fixationLoc is None:
        self.fixationLoc = self.maxSalienceLoc
        self.fixationSlice = np.index_exp[int(self.fixationLoc[1] - self.foveaSize[1] / 2):int(self.fixationLoc[1] + self.foveaSize[1] / 2), int(self.fixationLoc[0] - self.foveaSize[0] / 2):int(self.fixationLoc[0] + self.foveaSize[0] / 2)]
        # NOTE: This slice could be smaller than self.foveaSize
        self.logger.info("Fixated at: {}, fixation slice: {}".format(self.fixationLoc, self.fixationSlice))
      # Update feature vector representing current state of neurons
      self.logger.debug("[{:.2f}] Features: {}".format(self.timeNow, self.featureVector))  # [verbose]
      #self.logger.debug("[{:.2f}] Feature matrix:\n  {}".format(self.timeNow, "\n  ".join("{}: {}".format(label, self.featureMatrix[i]) for i, label in enumerate(self.featureLabels))))  # [very verbose!]
      self.buffers['features'].set_out(dict(izip(self.featureLabels, self.featureVector)))  # TODO: find a better way than zipping every iteration (named tuple or something?)
      if self.timeNow > (self.lastTransitionTime + self.min_fixation_duration):
        # TODO: Update match buffer based on feature values and weights
        # TODO: Compute utility based on duration of fixation (falling activation), match and/or salience
        # TODO: If very high utility, turn on hold (assuming agent will ask us to release)
        #       If low utility or past max_fixation_duration, switch to FREE state and look somewhere else
        maxSalienceLocDist = hypot(self.maxSalienceLoc[0] - self.fixationLoc[0], self.maxSalienceLoc[1] - self.fixationLoc[1])
        
        # Put a limit on hold
        if self.hold and self.timeNow > (self.lastTransitionTime + self.max_hold_duration):
          self.hold = False  # NOTE: This forcefully breaks a hold; might be better to depend on salient stimuli
        
        # Check for possible transitions out of FIXATE
        if not self.hold and \
            (maxSalienceLocDist > self.fovealRadius or \
             self.maxSalience < self.saccadeSalience or \
             self.timeNow > (self.lastTransitionTime + self.max_fixation_duration)):
          # Create FINST to inhibit current location in future, before switching to FREE
          if self.maxSalience >= self.min_saccade_salience:  # if current location is still salient enough to elicit a saccade
            self.finsts.append(Finst(self.fixationLoc, self.ocularMotionSystem.getFocusPoint(), timeCreated=self.timeNow))  # TODO: pass in activationCreated once FINSTs are stored in priority queue
          self.fixationLoc = None  # set to None to indicate we're no longer fixated; next fixation will store a new location
          self.transition(self.State.FREE)
    
    # * Show output images if in GUI mode
    if self.context.options.gui:
      #cv2.imshow("Hue", self.images['H'])
      #cv2.imshow("Saturation", self.images['S'])
      #cv2.imshow("Value", self.images['V'])
      if self.context.options.debug:  # only show detail when in debug mode; limit to important images/maps
        #cv2.imshow("Rod response", self.images['Rod'])
        #for coneType, coneImage in self.images['Cone'].iteritems():
        #  cv2.imshow("{} Cones".format(coneType), coneImage)
        for bipolarType, bipolarImage in self.images['Bipolar'].iteritems():
          cv2.imshow("{} Bipolar cells".format(bipolarType), bipolarImage)
        for ganglionType, ganglionImage in self.images['Ganglion'].iteritems():
          cv2.imshow("{} Ganglion cells".format(ganglionType), ganglionImage)
          #cv2.imshow("{} Ganglion cells".format(ganglionType), np.sqrt(self.featurePathways[ganglionType].p) * ganglionImage)  # show image weighted by selected feature probability, artificially scaled to make responses visible
      #cv2.imshow("Salience", self.images['Salience'])  # combined salience image
      
      # Designate a representative output image
      #self.imageOut = cv2.bitwise_and(self.retina.images['BGR'], self.retina.images['BGR'], mask=self.imageSelectionOut)  # mask out everything outside selected neuron's receptive field
      self.imageOut = self.images['Salience']  # make a copy?
      #_, self.imageOut = cv2.threshold(self.imageOut, 0.5, 1.0, cv2.THRESH_TOZERO)  # apply threshold to remove low-response regions
      self.imageOut = np.uint8(self.imageOut * 255)  # convert to uint8 image for display (is this necessary?)
      if self.maxSalience >= self.min_saccade_salience:
        cv2.circle(self.imageOut, self.maxSalienceLoc, 3, 175, -1)  # mark most salient location with a small faint dot
        if self.maxSalience >= self.min_good_salience:
          cv2.circle(self.imageOut, self.maxSalienceLoc, int(self.maxSalience * 25), int(128 + self.maxSalience * 127), 1 + int(self.maxSalience * 4))  # highlight highly salient locations: larger, fatter, brighter for higher salience value
      if self.state == self.State.FIXATE and self.fixationLoc is not None:
        cv2.circle(self.imageOut, self.fixationLoc, 1, 225, -1)  # mark fixation location with a tiny bright dot
      cv2.putText(self.imageOut, self.State.toString(self.state) + (" (holding)" if self.hold else ""), (20, 30), cv2.FONT_HERSHEY_PLAIN, 1, 200)  # show current state
    
    return True, self.imageOut
  
  def stop(self):
    # TODO Ensure this gets called for proper clean-up, esp. now that we are using an animated plot
    if self.context.options.gui:
      self.neuronPotentialMonitor.stop()
  
  def transition(self, next_state):
    self.logger.info("[{:.2f}] Transitioning from {} to {} state after {:.2f}s".format(self.timeNow, self.State.toString(self.state), self.State.toString(next_state), (self.timeNow - self.lastTransitionTime)))
    self.state = next_state
    self.lastTransitionTime = self.timeNow
    self.buffers['state'].set_out(self.state)  # update corresponding buffer
  
  def handleIntent(self, intent):
    if intent is None or intent not in self.intents:
      self.logger.warning("Unknown/null intent: '%s'", intent)
      return
    
    self.logger.info("Intent: %s", intent)
    if intent == 'find':
      # NOTE All relevant buffers must be set *before* find intent is sent in
      self.transition(self.State.FREE)  # reset state to use new weights
      self.hold = False  # implies we can move around again
    elif intent == 'hold':
      self.hold = True  # system won't perform saccades, even if utility drops
      if self.state == self.State.FREE:
        self.transition(self.State.FIXATE)  # transition to FIXATE state (unless performing a saccade)
    elif intent == 'release':
      self.hold = False  # system can resume FIXATE-SACCADE cycle
    elif intent == 'reset':
      self.finsts.clear()
      self.transition(self.State.SACCADE)
      self.ocularMotionSystem.reset()  # reset to the center of visual stream
      self.hold = False
    else:
      self.logger.warning("Unhandled intent: '%s'", intent)
  
  def performSaccade(self, saccadeTarget=None):
    if self.ocularMotionSystem is not None:
      self.transition(self.State.SACCADE)
      if saccadeTarget is not None:
        self.ocularMotionSystem.move(saccadeTarget)
      else:
        self.ocularMotionSystem.reset()
    else:
      self.logger.warning("Ocular motion system not found, skipping to FIXATE")
      self.transition(self.State.FIXATE)
  
  def inhibitMapAtFinst(self, imageMap, finst):
    loc = finst.getAdjustedLocation(self.ocularMotionSystem.getFocusPoint())
    cv2.circle(imageMap, loc, finst.radius, 0.0, cv.CV_FILLED)
    #cv2.putText(imageMap, "{:.2f}".format(finst.timeCreated), (loc[0] + finst.radius, loc[1] - finst.radius), cv2.FONT_HERSHEY_PLAIN, 1, 0.0)  # [debug]
    # TODO: Soft inhibition using finst.activation?
  
  def updateFeatureWeights(self, featureWeights, rest=None):
    """Update weights for features mentioned in given dict, using rest for others if not None."""
    # TODO Handle special labels for spatial selection
    if rest is None:
      rest = featureWeights.get('rest', None)  # rest may also be passed in as a dict item
    for label, pathway in self.featurePathways.iteritems():
      if label in featureWeights:
        pathway.p = featureWeights[label]
      elif rest is not None:
        pathway.p = rest
  
  def updateFeatureVector(self):
    # TODO: Also compute mean and variance over a moving window here? (or should that be an agent/manager-level function?)
    # Feature vector picks a single value from each channel
    self.featureVector = np.float32([pathway.output.neurons[0].potential for pathway in self.featurePathways.itervalues()])
    # Feature matrix picks all neuron values from each channel
    self.featureMatrix = np.float32([[neuron.potential for neuron in pathway.output.neurons] for pathway in self.featurePathways.itervalues()])
  
  def toCenterRelative(self, coords):
    return (coords[0] - self.imageCenter[0], coords[1] - self.imageCenter[1])  # convert to center-relative coordinates
  
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
  
  def createRetina(self):
    # TODO * Create Photoreceptor layer
    # TODO * Create BipolarCell layer
    # TODO * Create GanglionCell layer
    pass
  
  def createVisualCortex(self):
    # * Create several feature pathways, each with a salience, selection and feature layer
    self.featureLabels = self.images['Ganglion'].keys()  # cached for frequent use (NOTE currently will need to be updated if self.images['Ganglion'] changes)
    self.featurePathways = OrderedDict()
    for pathwayLabel in self.featureLabels:  # Ganglion cells are the source of each low-level visual pathway
      self.logger.info("Creating '{}' feature pathway".format(pathwayLabel))
      # ** Create layers
      # *** Salience neurons (TODO introduce magno and parvo types; expose layer parameters such as Z-axis position)
      salienceLayerBounds = np.float32([[0.0, 0.0, 0.0], [self.imageSize[0] - 1, self.imageSize[1] - 1, 0.0]])
      salienceNeuronDistribution = MultivariateNormal(mu=self.center, cov=(np.float32([self.center[0] ** 1.25, self.center[1] ** 1.25, 1.0]) * np.identity(3, dtype=np.float32)))
      #salienceNeuronDistribution = MultivariateUniform(lows=[0.0, 0.0, 0.0], highs=[self.imageSize[0], self.imageSize[1], 0.0])
      salienceNeurons = Population(numNeurons=self.num_salience_neurons, timeNow=self.timeNow, neuronTypes=[SalienceNeuron], bounds=salienceLayerBounds, distribution=salienceNeuronDistribution, system=self, pathway=pathwayLabel, imageSet=self.images['Ganglion'])
      # TODO self.addPopulation(salienceNeurons)?
      
      # *** Selection neurons
      selectionLayerBounds = np.float32([[0.0, 0.0, 50.0], [self.imageSize[0] - 1, self.imageSize[1] - 1, 50.0]])
      selectionNeuronDistribution = MultivariateNormal(mu=self.center + np.float32([0.0, 0.0, 50.0]), cov=(np.float32([self.center[0] ** 1.25, self.center[1] ** 1.25, 1.0]) * np.identity(3, dtype=np.float32)))
      #selectionNeuronDistribution = MultivariateUniform(lows=[0.0, 0.0, 50.0], highs=[self.imageSize[0], self.imageSize[1], 50.0])
      selectionNeurons = Population(numNeurons=self.num_selection_neurons, timeNow=self.timeNow, neuronTypes=[SelectionNeuron], bounds=selectionLayerBounds, distribution=selectionNeuronDistribution, system=self, pathway=pathwayLabel)
      # TODO self.addPopulation(selectionNeurons)?
      
      # *** Feature neurons (usually a single neuron for most non spatially-sensitive features)
      featureLayerBounds = np.float32([[0.0, 0.0, 100.0], [self.imageSize[0] - 1, self.imageSize[1] - 1, 100.0]])
      featureNeuronDistribution = MultivariateNormal(mu=self.center + np.float32([0.0, 0.0, 100.0]), cov=(np.float32([self.center[0] / 10, self.center[1] / 10, 1.0]) * np.identity(3, dtype=np.float32)))  # positioning doesn't matter much
      featureNeurons = Population(numNeurons=self.num_feature_neurons, timeNow=self.timeNow, neuronTypes=[FeatureNeuron], bounds=featureLayerBounds, distribution=featureNeuronDistribution, system=self, pathway=pathwayLabel)
      # TODO Set feature neuron plotColor to something more representative of the pathway
      
      # ** Connect neuron layers
      # *** Salience neurons to selection neurons (TODO use createProjection() once Projection is implemented, and register using self.addProjection)
      salienceNeurons.connectWith(selectionNeurons, maxConnectionsPerNeuron=5)
      # For selection neurons, finalize their receptive field radii based on connected neurons (average distance to extrema)
      minRFRadius = None
      maxRFRadius = None
      for selectionNeuron in selectionNeurons.neurons:
        xlim = [selectionNeuron.location[0], selectionNeuron.location[0]]  # min, max
        ylim = [selectionNeuron.location[1], selectionNeuron.location[1]]  # min, max
        for inputNeuron in selectionNeuron.inputNeurons:
          xlim[0] = min(xlim[0], inputNeuron.location[0] - inputNeuron.rfRadius)
          xlim[1] = max(xlim[1], inputNeuron.location[0] + inputNeuron.rfRadius)
          ylim[0] = min(ylim[0], inputNeuron.location[1] - inputNeuron.rfRadius)
          ylim[1] = max(ylim[1], inputNeuron.location[1] + inputNeuron.rfRadius)
        selectionNeuron.rfRadius = int((hypot(xlim[0] - selectionNeuron.location[0], ylim[0] - selectionNeuron.location[1]) + \
                                        hypot(xlim[1] - selectionNeuron.location[0], ylim[1] - selectionNeuron.location[1])) / 2)
        # NOTE: We don't need much precision for this estimated RF radius - it is mainly used to categorize these neurons into broad groups, and for display
        if minRFRadius is None or selectionNeuron.rfRadius < minRFRadius:
          minRFRadius = selectionNeuron.rfRadius
        if maxRFRadius is None or selectionNeuron.rfRadius > maxRFRadius:
          maxRFRadius = selectionNeuron.rfRadius
      
      # *** Selection neurons to feature neurons (all-to-all, filtered by receptive field size)
      featureRFRadiusStep = float(maxRFRadius - minRFRadius) / self.num_feature_neurons  # size of each uniform RF radius division to categorize input neurons in the featureNeurons layer
      for source in selectionNeurons.neurons:
        # All-to-all
        #for target in featureNeurons.neurons:
        #  source.synapseWith(target)
        # Filtered by receptive field size
        idx = int((source.rfRadius - minRFRadius) / featureRFRadiusStep)
        if idx >= self.num_feature_neurons:
          idx = self.num_feature_neurons - 1  # ensure idx is range
        source.synapseWith(featureNeurons.neurons[idx])  # connect with appropriate feature neuron
      selectionNeurons.isConnected = True  # NOTE need to explicitly do this since we're not using Population.connectWith()
      
      # *** Selection neurons to themselves (lateral inhibition; TODO make this a re-entrant inhibitory Projection with allow_self_connections=False?)
      for source in selectionNeurons.neurons:
        for target in selectionNeurons.neurons:
          if source == target: continue
          source.gateNeuron(target)
      
      # ** Add to dictionary of feature pathways
      self.featurePathways[pathwayLabel] = VisualFeaturePathway(label=pathwayLabel, populations=[salienceNeurons, selectionNeurons, featureNeurons], projections=None, output=featureNeurons, timeNow=self.timeNow)
  
      # ** Show neuron layers and connections [debug]
      #plotPopulations([salienceNeurons, selectionNeurons, featureNeurons], showConnections=True, equalScaleZ=True)  # [debug]
    
    # * Initialize feature vector
    self.featureVector = None
    self.updateFeatureVector()
  
  @rpc.enable
  def getBuffer(self, name):
    try:
      value = self.buffers[name].get()
      if callable(value):  # allows output buffer values to be callables (e.g. getter functions) that get called when retrieved
        value = value()
      #self.logger.debug("%s: %s", name, value)  # [verbose]
      return value
    except KeyError as e:
      self.logger.error("Buffer KeyError: %s", e)
    except BufferAccessError as e:
      self.logger.error("BufferAccessError (get '%s'): %s", name, e)
    return None  # failed
  
  @rpc.enable
  def setBuffer(self, name, value):
    try:
      #self.logger.debug("%s: %s", name, value)  # [verbose]
      obj = self.buffers[name].value  # NOTE direct access (not encouraged - can this be done using simple Python properties?)
      if callable(obj):  # allows input buffer values to be callables (e.g. setter functions) that get called when the buffer is written to
        obj(value)
      else:
        self.buffers[name].set(value)
      return True  # NOTE may not give the right indication if obj was a callable and returned a meaningful value
    except KeyError as e:
      self.logger.error("Buffer KeyError: %s", e)
    except BufferAccessError as e:
      self.logger.error("BufferAccessError (set '%s'): %s", name, e)
    return False  # failed
  
  @rpc.enable
  def listBuffers(self, types=False):
    """Return a list of exposed buffers (flat list), optionally with each buffer's type as well (list of 2-tuples)."""
    return [(name, buf.__class__.__name__) if types else name for name, buf in self.buffers.iteritems()]
  
  @rpc.enable_image
  def getImage(self, key='BGR'):
    try:
      return self.images[key]
    except KeyError as e:
      self.logger.error("Image KeyError: %s", e)
    return None
  
  @rpc.enable_image
  def getFovealImage(self, key='BGR'):
    try:
      return self.images[key][self.fovealSlice]
    except KeyError as e:
      self.logger.error("Image KeyError: %s", e)
    return None
  
  @rpc.enable_image
  def getFixatedImage(self, key='BGR'):
    try:
      return self.images[key][self.fixationSlice]
    except KeyError as e:
      self.logger.error("Image KeyError: %s", e)
    return None
  
  @rpc.enable_image
  def getOutputImage(self):
    if self.context.options.gui:
      return self.imageOut
    else:
      return None


class VisionManager(Projector):
  """A version of Projector that defaults to using a VisualSystem as target."""
  
  def __init__(self, target=None, *args, **kwargs):
    Projector.__init__(self, target if target is not None else VisualSystem(), *args, **kwargs)
    self.visualSystem = self.target  # synonym - Projector uses the generic term target
    self.ocularMotionSystem = EmulatedOcularMotionSystem(self, timeNow=self.context.timeNow)
    self.visualSystem.ocularMotionSystem = self.ocularMotionSystem
  
  def process(self, imageIn, timeNow):
    self.ocularMotionSystem.update(timeNow)
    return Projector.process(self, imageIn, timeNow)


class FeatureManager(VisionManager):
  """A visual system manager for computing stable features."""
  
  State = Enum(('NONE', 'INCOMPLETE', 'UNSTABLE', 'STABLE'))
  min_duration_incomplete = 2.0  # min. seconds to spend in incomplete state before transitioning (rolling buffer not full yet/neurons not activated enough)
  min_duration_unstable = 2.0  # min. seconds to spend in unstable state before transitioning (avoid short stability periods)
  max_duration_unstable = 5.0  # max. seconds to spend in unstable state before transitioning (avoid being stuck waiting forever for things to stabilize)
  min_duration_stable = 0.5  # avoid quick switches (attention deficiency)
  max_duration_stable = 2.0  # don't stare for too long (excess fixation)
  feature_buffer_size = 10  # number of iterations/samples to compute feature vector statistics over (rolling window)
  max_feature_sd = 0.005  # max. s.d. (units: Volts) to tolerate in judging a signal as stable
  
  def __init__(self, *args, **kwargs):
    kwargs['screen_background'] = kwargs.get('screen_background', np.uint8([0, 0, 0]))
    VisionManager.__init__(self, *args, **kwargs)
    self.state = self.State.NONE
    self.lastTransitionTime = -1.0
  
  def initialize(self, imageIn, timeNow):
    VisionManager.initialize(self, imageIn, timeNow)
    self.numFeatures = len(self.visualSystem.featureVector)
    self.featureVectorBuffer = np.zeros((self.feature_buffer_size, self.numFeatures), dtype=np.float32)  # rolling buffer of feature vector samples
    self.featureVectorIndex = 0  # index into feature vector buffer (count module size)
    self.featureVectorCount = 0  # no. of feature vector samples collected (same as index, sans modulo)
    self.featureVectorMean = np.zeros(self.numFeatures, dtype=np.float32)  # column mean of values in buffer
    self.featureVectorSD = np.zeros(self.numFeatures, dtype=np.float32)  # standard deviation of values in buffer
    self.featureMatrixBuffer = np.zeros((self.feature_buffer_size, self.numFeatures, self.visualSystem.num_feature_neurons), dtype=np.float32)  # follows featureVectorBuffer
    self.featureMatrixMean = np.zeros((self.numFeatures, self.visualSystem.num_feature_neurons), dtype=np.float32)  # follows featureVectorMean
    self.logger.info("[{:.2f}] Features: {}".format(timeNow, self.visualSystem.featureLabels))
    self.transition(self.State.INCOMPLETE, timeNow)
    self.logger.debug("Initialized")
  
  def process(self, imageIn, timeNow):
    keepRunning, imageOut = VisionManager.process(self, imageIn, timeNow)
    
    # Compute featureVector mean and variance over a moving window (also featureMatrix mean)
    self.featureVectorBuffer[self.featureVectorIndex, :] = self.visualSystem.featureVector
    self.featureMatrixBuffer[self.featureVectorIndex, :] = self.visualSystem.featureMatrix
    self.featureVectorCount += 1
    self.featureVectorIndex = self.featureVectorCount % self.feature_buffer_size
    np.mean(self.featureVectorBuffer, axis=0, dtype=np.float32, out=self.featureVectorMean)  # always update mean, in case someone needs it
    # TODO: debug here
    np.mean(self.featureMatrixBuffer, axis=0, dtype=np.float32, out=self.featureMatrixMean)
    
    # Change state according to feature vector values (and visual system's state)
    deltaTime = timeNow - self.lastTransitionTime
    if self.state == self.State.INCOMPLETE and \
       deltaTime > self.min_duration_incomplete and \
       self.featureVectorCount >= self.feature_buffer_size and \
       self.visualSystem.state == VisualSystem.State.FIXATE:
      self.visualSystem.setBuffer('intent', 'hold')  # ask system to hold gaze (i.e. no saccades)
      self.transition(self.State.UNSTABLE, timeNow)
    elif self.state == self.State.UNSTABLE or self.state == self.State.STABLE:
      if self.visualSystem.state == VisualSystem.State.FIXATE:
        np.std(self.featureVectorBuffer, axis=0, dtype=np.float32, out=self.featureVectorSD)
        self.logger.debug("[{:.2f}] Mean: {}".format(timeNow, self.featureVectorMean))  # [verbose]
        self.logger.debug("[{:.2f}] S.D.: {}".format(timeNow, self.featureVectorSD))  # [verbose]
        self.logger.debug("[{:.2f}] Feature matrix:\n  {}".format(timeNow, "\n  ".join("{}: {}".format(label, self.featureMatrixMean[i]) for i, label in enumerate(self.visualSystem.featureLabels))))
        if self.state == self.State.UNSTABLE and deltaTime > self.min_duration_unstable and \
            (np.max(self.featureVectorSD) <= self.max_feature_sd or deltaTime > self.max_duration_unstable):  # TODO use a time-scaled low-pass filtered criteria
          self.transition(self.State.STABLE, timeNow)
        elif self.state == self.State.STABLE and deltaTime > self.min_duration_stable and \
            (np.max(self.featureVectorSD) > self.max_feature_sd or deltaTime > self.max_duration_stable):
            self.transition(self.State.UNSTABLE, timeNow)
            self.visualSystem.setBuffer('intent', 'find')  # let system return to FIXATE-SACCADE mode (without inhibition)
      else:  # something made visual system lose focus, including us releasing the system
        self.transition(self.State.INCOMPLETE, timeNow)
    
    return keepRunning, imageOut
  
  def transition(self, next_state, timeNow):
    self.logger.debug("[{:.2f}] Transitioning from {} to {} state after {:.2f}s".format(timeNow, self.State.toString(self.state), self.State.toString(next_state), (timeNow - self.lastTransitionTime)))
    self.state = next_state
    self.lastTransitionTime = timeNow
  
  @rpc.enable
  def getState(self):
    return self.State.toString(self.state)
  
  @rpc.enable
  def getFeatureVector(self):
    return self.featureVectorMean.tolist()
  
  @rpc.enable
  def getFeatureMatrix(self):
    return self.featureMatrixMean.tolist()  # will be a nested list, not flat


def main(managerType=VisionManager):
  """Run end-to-end visual system."""
  
  context = Context.createInstance(description="Run a VisualSystem instance using a {}".format(managerType.__name__))
  print "main(): Creating visual system and manager"
  visSystem = VisualSystem()
  visManager = managerType(visSystem)
  
  if context.isRPCEnabled:
    print "main(): Exporting RPC calls"
    rpc.export(visSystem)
    rpc.export(visManager)
    rpc.refresh()  # Context is expected to have started RPC server
  
  print "main(): Starting vision loop"
  run(visManager)
  
  if context.isRPCEnabled:
    rpc.stop_server()  # do we need to do this if server is running as a daemon?
  print "main(): Done."


def test_FeatureManager_RPC():
  from time import sleep
  from multiprocessing import Process, Value
  
  Context.createInstance()
  print "test_FeatureManager_RPC(): Creating visual system and manager"
  visSystem = VisualSystem()
  visManager = FeatureManager(visSystem)
  
  print "test_FeatureManager_RPC(): Exporting RPC calls"
  rpc.export(visSystem)  # order of export vs. enable doesn't matter - everything will be resolved in refresh(), called by start_server()
  rpc.export(visManager)
  
  print "test_FeatureManager_RPC(): Starting RPC server thread"
  rpcServerThread = rpc.start_server_thread(daemon=True)
  
  # NOTE shared_loop_flag must be a multiprocessing.Value or .RawValue
  # NOTE gui should be set to true only if this is being run in its own dedicated process, without any shared GUI infrastructure
  def rpcClientLoop(shared_loop_flag, gui=False):
    with rpc.Client() as rpcClient:
      while shared_loop_flag.value == 1:
        try:
          for call in ['FeatureManager.getState', 'FeatureManager.getFeatureVector']:  # 'VisualSystem.getOutputImage'
            print "[RPC-Client] REQ:", call
            retval = rpcClient.call(call)
            if isinstance(retval, np.ndarray):
              print "[RPC-Client] REP[image]: shape: {}, dtype: {}".format(retval.shape, retval.dtype)
              # NOTE Qt (and possibly other backends) can only display from the main thread of a process
              if gui:
                cv2.imshow("VisualSystem output", retval)
                cv2.waitKey(10)
            else:
              print "[RPC-Client] REP:", retval
            if retval is None:
              break
            sleep(0.5)  # small sleep to prevent flooding
          sleep(0.5)  # extra sleep after each state, vector pair
        except KeyboardInterrupt:
          break
  
  print "test_FeatureManager_RPC(): Starting RPC client process"
  rpc_client_loop_flag = Value('i', 1)
  # NOTE No GUI output possible from child process; this will simply print metadata for any images received
  rpcClientProcess = Process(target=rpcClientLoop, name="RPC-Client", args=(rpc_client_loop_flag,))
  rpcClientProcess.daemon=True
  rpcClientProcess.start()
  sleep(0.01)  # let new process start
  
  print "test_FeatureManager_RPC(): Starting vision loop"
  run(visManager)
  print "test_FeatureManager_RPC(): Vision loop done; waiting for RPC threads/processes to join..."
  rpc_client_loop_flag.value = 0
  if rpc.Client.recv_timeout is not None:  # just a guess, actual timeout used could be different
    rpcClientProcess.join(rpc.Client.recv_timeout / 1000.0 + 1.0)
  print "test_FeatureManager_RPC(): RPC client process joined (or timeout)"
  rpc.stop_server()
  if rpc.Server.recv_timeout is not None:  # just a guess, actual timeout used could be different
    rpcServerThread.join(rpc.Server.recv_timeout / 1000.0 + 1.0)
  print "test_FeatureManager_RPC(): RPC server thread joined (or timeout)"
  print "test_FeatureManager_RPC(): Done."


# Testing
if __name__ == "__main__":
  # NOTE Defaults to using FeatureManager instead of VisualManager
  choices = [('--test_rpc', "Test RPC functionality by running a client, server pair")]
  context = Context.createInstance(parent_argparsers=[Context.createChoiceParser(choices)])
  if context.options.test_rpc:
    test_FeatureManager_RPC()
  else:
    main(managerType=FeatureManager)  # will enable RPC calls if --rpc was passed in
