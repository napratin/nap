"""A biologically-inspired model of visual perception."""

from math import exp
import logging
import numpy as np
import cv2
import cv2.cv as cv
from collections import OrderedDict

#import pyNN.neuron as sim
from lumos.context import Context
from lumos.util import Enum
from lumos.input import InputDevice, run

from ..neuron import Neuron, Population, Projection, neuron_inhibition_period, Uniform, MultivariateUniform, NeuronMonitor, plotPopulations
from .photoreceptor import Rod, Cone
from .simplified.retina import SimplifiedProjector
from .simplified.visual_cortex import SalienceNeuron, SelectionNeuron, FeatureNeuron


# Global variables
default_feature_weight = 0.75  # default weight for a feature pathway, treated as update probability for its neurons

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


class VisualSystem(object):
  """Complete system for processing dynamic visual input."""
  
  num_rods = 10000  # humans: 90-120 million
  num_cones = 1000  # humans: 4.5-6 million
  num_bipolar_cells = 2000
  num_ganglion_cells = 1000
  num_salience_neurons = 400
  num_selection_neurons = 100
  
  default_image_size = (256, 256)  # (width, height)
  
  def __init__(self, imageSize=default_image_size, timeNow=0.0, showMonitor=True):
    # * Get context and logger
    self.context = Context.getInstance()
    self.logger = logging.getLogger(self.__class__.__name__)
    
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
    
    # ** Spatial attention map with a central (covert) spotlight (currently unused; TODO move to VisualCortex? also, use np.ogrid?)
    #self.image['Attention'] = np.zeros(self.imageShapeC1, dtype=self.imageTypeFloat)
    #cv2.circle(self.image['Attention'], self.imageCenter, self.imageSize[0] / 3, 1.0, cv.CV_FILLED)
    #self.image['Attention'] = cv2.blur(self.image['Attention'], (self.imageSize[0] / 4, self.imageSize[0] / 4))  # coarse blur
    
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
    
    # * Output image and plots
    if self.context.options.gui:
      #self.imageOut = np.zeros(self.imageShapeC3, dtype=self.imageTypeInt)
      # TODO Salience and selection output will be for each feature pathway
      self.imageSalienceOut = np.zeros(self.imageShapeC1, dtype=self.imageTypeInt)  # salience neuron outputs
      self.imageSelectionOut = np.zeros(self.imageShapeC1, dtype=self.imageTypeInt)  # selection neuron outputs
      
      if showMonitor:
        self.neuronPotentialMonitor = NeuronMonitor()
        for pathwayLabel, featurePathway in self.featurePathways.iteritems():
          self.neuronPotentialMonitor.addChannel(label=pathwayLabel, obj=featurePathway.output.neurons[0], color=self.featurePlotColors[pathwayLabel])  # very hard-coded way to access single output neuron!
        self.neuronPotentialMonitor.start()
  
  def update(self, timeNow):
    self.timeNow = timeNow
    
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
    self.images['Bipolar']['OFF'] = np.clip((1.0 - self.images['Rod']) - 0.75 * (1.0 - imageRodBlurred), 0.0, 1.0)  # same as (1 - ON response)? (nope)
    
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
      self.images['Ganglion']['GR'] = np.maximum(self.images['Ganglion']['GR'], np.clip(cv2.filter2D(-imageRG, -1, k), 0.0, 1.0))
      self.images['Ganglion']['RB'] = np.maximum(self.images['Ganglion']['RB'], np.clip(cv2.filter2D(imageRB, -1, k), 0.0, 1.0))
      self.images['Ganglion']['BR'] = np.maximum(self.images['Ganglion']['BR'], np.clip(cv2.filter2D(-imageRB, -1, k), 0.0, 1.0))
      self.images['Ganglion']['BY'] = np.maximum(self.images['Ganglion']['BY'], np.clip(cv2.filter2D(imageBY, -1, k), 0.0, 1.0))
      self.images['Ganglion']['YB'] = np.maximum(self.images['Ganglion']['YB'], np.clip(cv2.filter2D(-imageBY, -1, k), 0.0, 1.0))
    
    # * Compute combined (salience) image; TODO incorporate attention weighting (spatial, as well as by visual feature)
    # ** Method 1: Max of all Ganglion cell images
    self.images['Salience'].fill(0.0)
    for ganglionType, ganglionImage in self.images['Ganglion'].iteritems():
      #self.images['Salience'] = np.maximum(self.images['Salience'], ganglionImage)
      #self.logger.debug("[Salience] Combining {}".format(self.featurePathways[ganglionType]))  # [verbose]
      self.images['Salience'] = np.maximum(self.images['Salience'], np.sqrt(self.featurePathways[ganglionType].p) * ganglionImage)  # take maximum, scaled by feature pathway probabilities (for display only)
      #self.images['Salience'] = self.images['Salience'] + (self.numGanglionTypes_inv * np.sqrt(self.featurePathways[ganglionType].p) * ganglionImage)  # take normalized sum (mixes up features), scaled by feature pathway probabilities (for display only)
    
    self.images['Salience'] = cv2.blur(self.images['Salience'], (3, 3))  # blur slightly to smooth out specs
    #self.images['Salience'] *= self.image['Attention']  # TODO evaluate if this is necessary
    _, self.maxSalience, _, self.maxSalienceLoc = cv2.minMaxLoc(self.images['Salience'])  # find out most salient location (from combined salience map)
    #self.logger.debug("Max. salience value: {:5.3f} @ {}".format(self.maxSalience, self.maxSalienceLo))  # [verbose]
    if self.maxSalience < 0.66:
      self.maxSalience = 0.0
    
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
        
        # ** Render output images and show them
        if self.context.options.gui:
          # *** Salience neurons
          self.imageSalienceOut.fill(0.0)
          for salienceNeuron in salienceNeurons.neurons:
            # Render salience neuron's receptive field with response-based pixel value (TODO cache int radii and pixel as tuple?)
            cv2.circle(self.imageSalienceOut, (salienceNeuron.pixel[0], salienceNeuron.pixel[1]), np.int_(salienceNeuron.rfRadius), 128)
            cv2.circle(self.imageSalienceOut, (salienceNeuron.pixel[0], salienceNeuron.pixel[1]), np.int_(salienceNeuron.rfCenterRadius), salienceNeuron.pixelValue, cv.CV_FILLED)
          
          # *** Selection neurons (TODO gather representative receptive field from source connections and cache it for use here, instead of using constant size?)
          if featurePathway.selectedNeuron is not None and (timeNow - featurePathway.selectedTime) < 3.0:
            #self.imageSelectionOut.fill(0.0)
            cv2.circle(self.imageSalienceOut, (featurePathway.selectedNeuron.pixel[0], featurePathway.selectedNeuron.pixel[1]), self.imageSize[0] / 20, int(255 * exp(featurePathway.selectedTime - timeNow)), 2)  # draw selected neuron with a shade that fades with time (on salience output image)
            #cv2.circle(self.imageSelectionOut, (featurePathway.selectedNeuron.pixel[0], featurePathway.selectedNeuron.pixel[1]), self.imageSize[0] / 20, int(255 * exp(featurePathway.selectedTime - timeNow)), cv.CV_FILLED)  # draw selected neuron with a shade that fades with time
          
          cv2.imshow("{} Salience".format(pathwayLabel), self.imageSalienceOut)
          #cv2.imshow("{} Selection".format(pathwayLabel), self.imageSelectionOut)
    
    # * Update feature vector representing current state of neurons
    self.updateFeatureVector() 
    self.logger.debug("[{:.2f}] Features: {}".format(self.timeNow, self.featureVector))
    
    # * TODO Compute feature vector of attended region
    
    # * TODO Final output image
    #self.imageOut = cv2.bitwise_and(self.retina.images['BGR'], self.retina.images['BGR'], mask=self.imageSelectionOut)
    
    # * Show output images if in GUI mode
    if self.context.options.gui:
      cv2.imshow("Retina", self.images['BGR'])
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
      cv2.imshow("Salience", self.images['Salience'])
      
      # Designate a representative output image
      self.imageOut = self.images['Salience']  # make a copy?
      if self.maxSalience >= 0.66:
        cv2.circle(self.imageOut, self.maxSalienceLoc, int(self.maxSalience * 25), int(128 + self.maxSalience * 127), 1 + int(self.maxSalience * 4))  # mark most salient location: larger, fatter, brighter for higher salience value
      _, self.imageOut = cv2.threshold(self.imageOut, 0.5, 1.0, cv2.THRESH_TOZERO)  # apply threshold to remove low-response regions
  
  def stop(self):
    # TODO Ensure this gets called for proper clean-up, esp. now that we are using an animated plot
    if self.context.options.gui:
      self.neuronPotentialMonitor.stop()
  
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
      #salienceNeuronDistribution = MultivariateNormal(mu=self.center, cov=(np.float32([self.center[0] ** 2, self.center[0] ** 2, 1.0]) * np.identity(3, dtype=np.float32)))
      salienceNeuronDistribution = MultivariateUniform(lows=[0.0, 0.0, 0.0], highs=[self.imageSize[0], self.imageSize[1], 0.0])
      salienceNeurons = Population(numNeurons=self.num_salience_neurons, timeNow=self.timeNow, neuronTypes=[SalienceNeuron], bounds=salienceLayerBounds, distribution=salienceNeuronDistribution, system=self, pathway=pathwayLabel, imageSet=self.images['Ganglion'])
      # TODO self.addPopulation(salienceNeurons)?
      
      # *** Selection neurons
      selectionLayerBounds = np.float32([[0.0, 0.0, 50.0], [self.imageSize[0] - 1, self.imageSize[1] - 1, 50.0]])
      selectionNeuronDistribution = MultivariateUniform(lows=[0.0, 0.0, 50.0], highs=[self.imageSize[0], self.imageSize[1], 50.0])
      selectionNeurons = Population(numNeurons=self.num_selection_neurons, timeNow=self.timeNow, neuronTypes=[SelectionNeuron], bounds=selectionLayerBounds, distribution=selectionNeuronDistribution, system=self, pathway=pathwayLabel)
      # TODO self.addPopulation(selectionNeurons)?
      
      # *** Feature neurons (usually a single neuron for most non spatially-sensitive features)
      featureLayerBounds = np.float32([[0.0, 0.0, 100.0], [self.imageSize[0] - 1, self.imageSize[1] - 1, 100.0]])
      featureNeurons = Population(numNeurons=1, timeNow=self.timeNow, neuronTypes=[FeatureNeuron], bounds=featureLayerBounds, neuronLocations=np.float32([[self.imageCenter[0], self.imageCenter[1], 100.0]]), system=self, pathway=pathwayLabel)  # effectively a single point in space is what we need for location
      # TODO Set feature neuron plotColor to something more representative of the pathway
      
      # ** Connect neuron layers
      # *** Salience neurons to selection neurons (TODO use createProjection() once Projection is implemented, and register using self.addProjection)
      salienceNeurons.connectWith(selectionNeurons, maxConnectionsPerNeuron=5)
      
      # *** Selection neurons to feature neurons (all-to-all)
      for source in selectionNeurons.neurons:
        for target in featureNeurons.neurons:
          source.synapseWith(target)
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
  
  def updateFeatureWeights(self, featureWeights, rest=None):
    """Update weights for features mentioned in given dict, using rest for others if not None."""
    # TODO Handle special labels for spatial selection
    for label, pathway in self.featurePathways.iteritems():
      if label in featureWeights:
        pathway.p = featureWeights[label]
      elif rest is not None:
        pathway.p = rest
  
  def updateFeatureVector(self):
    self.featureVector = np.float32([pathway.output.neurons[0].potential for pathway in self.featurePathways.itervalues()])
    # TODO Also compute mean and variance over a moving window here? (or should that be an agent/manager-level function?)
      
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
  """A version of Projector that uses a simplified Retina and layers from the visual cortex for visual processing."""
  
  def __init__(self, retina=None):
    SimplifiedProjector.__init__(self, retina if retina is not None else VisualSystem())


class FeatureManager(VisionManager):
  """A visual system manager for computing stable features."""
  
  State = Enum(('NONE', 'INCOMPLETE', 'UNSTABLE', 'STABLE'))
  min_duration_incomplete = 2.0  # min. seconds to spend in incomplete state before transitioning (rolling buffer not full yet/neurons not activated enough)
  min_duration_unstable = 2.0  # min. seconds to spend in unstable state before transitioning (avoid short stability periods)
  max_duration_unstable = 5.0  # max. seconds to spend in unstable state before transitioning (avoid being stuck waiting forever for things to stabilize)
  feature_buffer_size = 10  # number of iterations/samples to compute feature vector statistics over (rolling window)
  max_feature_sd = 0.005  # max. s.d. (units: Volts) to tolerate in judging a signal as stable
  
  def __init__(self, visualSystem=None):
    self.visualSystem = visualSystem if visualSystem is not None else VisualSystem()  # TODO remove this once VisionManager (or its ancestor caches visualSystem in __init__)
    VisionManager.__init__(self, self.visualSystem)
    self.state = self.State.NONE
    self.timeStateChange = -1.0
  
  def initialize(self, imageIn, timeNow):
    VisionManager.initialize(self, imageIn, timeNow)
    self.numFeatures = len(self.visualSystem.featureVector)
    self.featureVectorBuffer = np.zeros((self.feature_buffer_size, self.numFeatures), dtype=np.float32)  # rolling buffer of feature vector samples
    self.featureVectorIndex = 0  # index into feature vector buffer (count module size)
    self.featureVectorCount = 0  # no. of feature vector samples collected (same as index, sans modulo)
    self.featureVectorMean = np.zeros(self.numFeatures, dtype=np.float32)  # column mean of values in buffer
    self.featureVectorSD = np.zeros(self.numFeatures, dtype=np.float32)  # standard deviation of values in buffer
    self.logger.info("[{:.2f}] Features: {}".format(timeNow, self.visualSystem.featureLabels))
    self.transition(self.State.INCOMPLETE, timeNow)
    self.logger.debug("Initialized")
  
  def process(self, imageIn, timeNow):
    keepRunning, imageOut = VisionManager.process(self, imageIn, timeNow)
    
    # Compute featureVector mean and variance over a moving window
    self.featureVectorBuffer[self.featureVectorIndex, :] = self.visualSystem.featureVector
    self.featureVectorCount += 1
    self.featureVectorIndex = self.featureVectorCount % self.feature_buffer_size
    
    # Change state according to feature vector values
    deltaTime = timeNow - self.timeStateChange
    if self.state == self.State.INCOMPLETE and deltaTime > self.min_duration_incomplete and self.featureVectorCount >= self.feature_buffer_size:
      self.transition(self.State.UNSTABLE, timeNow)
    elif self.state == self.State.UNSTABLE or self.state == self.State.STABLE:
      np.mean(self.featureVectorBuffer, axis=0, dtype=np.float32, out=self.featureVectorMean)
      np.std(self.featureVectorBuffer, axis=0, dtype=np.float32, out=self.featureVectorSD)
      self.logger.debug("[{:.2f}] Mean: {}".format(timeNow, self.featureVectorMean))
      self.logger.debug("[{:.2f}] S.D.: {}".format(timeNow, self.featureVectorSD))
      if self.state == self.State.UNSTABLE and deltaTime > self.min_duration_unstable and \
          (np.max(self.featureVectorSD) <= self.max_feature_sd or deltaTime > self.max_duration_unstable):  # TODO use a time-scaled low-pass filtered criteria
        self.transition(self.State.STABLE, timeNow)
      elif self.state == self.State.STABLE and np.max(self.featureVectorSD) > self.max_feature_sd:
        self.transition(self.State.UNSTABLE, timeNow)
    
    return keepRunning, imageOut
  
  def transition(self, next_state, timeNow):
    self.logger.debug("[{:.2f}] Transitioning from {} to {} state after {:.2f}s".format(timeNow, self.State.toString(self.state), self.State.toString(next_state), (timeNow - self.timeStateChange)))
    self.state = next_state
    self.timeStateChange = timeNow


def test_VisualSystem():
  # Test the end-to-end visual system
  Context.createInstance()
  run(VisionManager, description="Test application that uses a SimplifiedProjector to run image input through a VisualSystem instance.")


if __name__ == "__main__":
  test_VisualSystem()
