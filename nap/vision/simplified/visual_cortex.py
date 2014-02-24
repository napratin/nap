"""Simplified model of the visual cortex with appropriate neuron types."""

from math import exp, sqrt
import numpy as np
import cv2
import cv2.cv as cv

from lumos.input import run

from ...neuron import Neuron, NeuronGroup, threshold_potential, action_potential_peak, action_potential_trough, refractory_period, neuron_inhibition_period, Uniform, MultivariateUniform, MultivariateNormal, plotNeuronGroups
from .retina import Retina, SimplifiedProjector

class SalienceNeuron(Neuron):
  """A spiking neuron that responds to center-surround contrast in its receptive field."""
  
  _str_attrs = ['id', 'pixel', 'potential', 'pixelValue']
  
  # Receptive field parameters
  rf_radius_range = Uniform(low=0.025, high=0.2)  # fraction of visual field size
  rf_center_radius_range = Uniform(low=(1.0 / sqrt(2.0)) * 0.9, high=(1.0 / sqrt(2.0)) * 1.1)  # fraction of receptive field radius (NOTE in order to ensure equal center and surround area, center radius must be about 1/sqrt(2) of outer radius; otherwise use mean when computing center-surround difference)
  
  # TODO Define types of salience neurons (feature-specific, and feature-agnostic) as tuple of feature neurons/maps to connect to
  
  # Electrophysiological parameters for Integrate-and-Fire method (model)
  R = 300.0e06  # Ohms; membrane resistance (~30-700Mohm)
  C = 3.0e-09  # Farads; membrane capacitance (~2-3nF)
  tau = R * C  # seconds; time constant (~100-1000ms)
  max_current = 100.0e-12  # Amperes; determines how input response affects potential
  
  # Miscellaneous parameters
  potential_scale = 255 / abs(action_potential_peak - Neuron.resting_potential.mu)  # factor used to convert cell potential to image pixel value
  
  def __init__(self, location, timeNow, retina, pixel=None):
    Neuron.__init__(self, location, timeNow)
    self.retina = retina
    self.pixel = pixel if pixel is not None else np.int_(location[:2])
    self.expDecayFactor = 0.0
    self.response = 0.0
    self.I_e = 0.0
    self.pixelValue = 0
    
    # Receptive field parameters
    self.rfRadius = np.random.uniform(self.rf_radius_range.low, self.rf_radius_range.high) * self.retina.imageSize[0]
    self.rfCenterRadius = np.random.uniform(self.rf_center_radius_range.low, self.rf_center_radius_range.high) * self.rfRadius
    
    # Define receptive field as two slice/index expressions, one for the whole and one for center
    # NOTE Since whole = center + surround => surround = whole - center
    #    and, response = center - surrond
    #   thus, response = center - (whole - center) = 2 * center - whole
    self.rfSlice = np.index_exp[
      max(self.pixel[1] - self.rfRadius, 0):min(self.pixel[1] + self.rfRadius, self.retina.imageSize[1]),
      max(self.pixel[0] - self.rfRadius, 0):min(self.pixel[0] + self.rfRadius, self.retina.imageSize[0])]
    self.rfCenterSlice = np.index_exp[
      max(self.pixel[1] - self.rfCenterRadius, 0):min(self.pixel[1] + self.rfCenterRadius, self.retina.imageSize[1]),
      max(self.pixel[0] - self.rfCenterRadius, 0):min(self.pixel[0] + self.rfCenterRadius, self.retina.imageSize[0])]
  
  def updatePotential(self):
    # Gather response from feature map (TODO choose feature map based on type)
    rfWholeResponse = np.mean(self.retina.imageSalience[self.rfSlice])  # TODO check if mean() is appropriate here, or is sum() better (then we need to ensure +ve and -ve areas are roughly equal)?
    rfCenterResponse = np.mean(self.retina.imageSalience[self.rfCenterSlice])
    self.response = np.clip(2 * rfCenterResponse - rfWholeResponse, 0.0, 1.0)  # response = 2 * center - whole
    
    # Update potential: Differential equation solution (similar to Photoreceptor, Method 4)
    self.expDecayFactor = exp(-self.deltaTime / self.tau)
    self.I_e = self.max_current * self.response
    self.potential = self.resting_potential.mu + ((self.potentialLastUpdated - self.resting_potential.mu) * self.expDecayFactor) + (self.R * self.I_e * (1.0 - self.expDecayFactor))  # V(t) = V_r + ((V(t') - V_r) * (e ^ (-(t - t') / tau))) + (R * I_e * (1 - e ^ (-(t - t') / tau)))
    
    # Check for action potential event (TODO implement action potential using self-depolarizing current)
    if self.potential > threshold_potential and (self.timeCurrent - self.timeLastFired) >= refractory_period:
      self.actionPotential()
    
    # Compute a value to render (TODO report potential here for logging and plotting)
    self.pixelValue = int(np.clip(abs(self.potential - self.resting_potential.mu) * self.potential_scale, 0, 255))
    
    # Fire action potential, if we've reached peak
    #if self.id == 10: print "[{:.2f}] Potential: {}".format(self.timeCurrent, self)  # [debug]
    if self.potential >= action_potential_peak:
      #print "[{:.2f}] Action potential: {}".format(self.timeCurrent, self)  # [debug]
      self.fireActionPotential()
      self.timeLastFired = self.timeCurrent
      self.potential = np.random.normal(action_potential_trough.mu, action_potential_trough.sigma)  # repolarization/falling phase (instantaneous)
      #if self.id == 10: print "[{:.2f}] Post-action potential: {}".format(self.timeCurrent, self)  # [debug]


class SelectionNeuron(Neuron):
  """A spiking neuron that uses lateral inhibition as a selection mechanism."""
  
  _str_attrs = ['id', 'pixel', 'potential', 'pixelValue']
  
  # Electrophysiological parameters for Integrate-and-Fire method (model)
  R = 300.0e06  # Ohms; membrane resistance (~30-700Mohm)
  C = 3.0e-09  # Farads; membrane capacitance (~2-3nF)
  tau = R * C  # seconds; time constant (~100-1000ms)
  max_current = 120.0e-12  # Amperes; determines how input response affects potential
  
  # Miscellaneous parameters
  potential_scale = 255 / abs(action_potential_peak - Neuron.resting_potential.mu)  # factor used to convert cell potential to image pixel value
  
  def __init__(self, location, timeNow, retina, pixel=None):
    Neuron.__init__(self, location, timeNow)
    self.retina = retina
    self.pixel = pixel if pixel is not None else np.int_(location[:2])
    #self.expDecayFactor = 0.0
    self.pixelValue = 0
  
  def updatePotential(self):
    # Decay potential
    #self.expDecayFactor = exp(-self.deltaTime / self.tau)
    #self.potential = self.resting_potential.mu + ((self.potentialLastUpdated - self.resting_potential.mu) * self.expDecayFactor)  # exponential decay: V(t) = V_r + ((V(t') - V_r) * (e ^ (-(t - t') / tau)))
    self.potential -= (self.potentialLastUpdated - self.resting_potential.mu) * self.deltaTime / self.tau  # linear approximation of exponential decay: V(t) = V(t') - (V(t') - V_r) * ((t - t') / tau)
    # NOTE Linear approximation may cause underflow, but that shouldn't cause any problems
    
    # Accumulate/integrate incoming potentials
    self.potential += self.potentialAccumulated  # integrate signals accumulated from neighbors
    self.potentialAccumulated = 0.0  # reset accumulator (don't want to double count!)
    
    # Check for action potential event (TODO implement action potential using self-depolarizing current)
    if self.potential > threshold_potential and (self.timeCurrent - self.timeLastFired) >= refractory_period:
      self.actionPotential()
    
    # Compute representative pixel value to render (TODO report potential here for logging and plotting)
    self.pixelValue = int(np.clip(abs(self.potential - self.resting_potential.mu) * self.potential_scale, 0, 255))
    
    # Fire action potential, if we've reached peak
    if self.potential >= action_potential_peak:
      #print "[{:.2f}] Action potential: {}".format(self.timeCurrent, self)  # [debug]
      self.fireActionPotential()  # inhibits gated neurons
      self.timeLastFired = self.timeCurrent
      self.potential = np.random.normal(action_potential_trough.mu, action_potential_trough.sigma)  # repolarization/falling phase (instantaneous)


# TODO Define VisualCortex class that would contain the different cortical layers and expose an interface to the visual system (with retina passed in through context? or context.systems/context.components?)


class CorticalProjector(SimplifiedProjector):
  """A SimplifiedProjector-based type that runs visual input through a simplified Retina as well as higher cortical layers."""
  
  num_salience_neurons = 400
  num_selection_neurons = 100
  
  def __init__(self, retina=None):
    SimplifiedProjector.__init__(self, retina if retina is not None else Retina())
    
    # * Create layers of cortical neurons (TODO move this to VisualCortex; rename e.g. salienceNeurons -> populations['Salience'] for salience population, and with that neuron.NeuronGroup -> neuron.Population)
    # ** Salience neurons (TODO introduce magno and parvo types)
    self.salienceLayerBounds = np.float32([[0.0, 0.0, 0.0], [self.retina.imageSize[0] - 1, self.retina.imageSize[1] - 1, 0.0]])
    #self.salienceNeuronDistribution = MultivariateNormal(mu=self.retina.center, cov=(np.float32([self.retina.center[0] ** 2, self.retina.center[0] ** 2, 1.0]) * np.identity(3, dtype=np.float32)))
    self.salienceNeuronDistribution = MultivariateUniform(lows=[0.0, 0.0, 0.0], highs=[self.retina.imageSize[1], self.retina.imageSize[0], 0.0])
    self.salienceNeurons = NeuronGroup(numNeurons=self.num_salience_neurons, timeNow=0.0, neuronTypes=[SalienceNeuron], bounds=self.salienceLayerBounds, distribution=self.salienceNeuronDistribution, retina=self.retina)  # TODO use timeNow when moved to VisualCortex
    self.salienceNeuronPlotColor = 'coral'
    
    # ** Selection neurons
    self.selectionLayerBounds = np.float32([[0.0, 0.0, 50.0], [self.retina.imageSize[0] - 1, self.retina.imageSize[1] - 1, 50.0]])
    self.selectionNeuronDistribution = MultivariateUniform(lows=[0.0, 0.0, 50.0], highs=[self.retina.imageSize[1], self.retina.imageSize[0], 50.0])
    self.selectionNeurons = NeuronGroup(numNeurons=self.num_selection_neurons, timeNow=0.0, neuronTypes=[SelectionNeuron], bounds=self.selectionLayerBounds, distribution=self.selectionNeuronDistribution, retina=self.retina)  # TODO use timeNow when moved to VisualCortex
    self.selectionNeuronPlotColor = 'olive'
    
    # * Connect neuron layers
    # ** Salience neurons to selection neurons
    self.salienceNeurons.connectWith(self.selectionNeurons, maxConnectionsPerNeuron=5)
    
    # ** Selection neurons to themselves (lateral inhibition; TODO make this a function in Projection)
    for source in self.selectionNeurons.neurons:
      for target in self.selectionNeurons.neurons:
        if source == target: continue
        source.gateNeuron(target)
    
    # * Show neuron layers and connections [debug]
    #plotNeuronGroups([self.salienceNeurons, self.selectionNeurons], groupColors=[self.salienceNeuronPlotColor, self.selectionNeuronPlotColor], showConnections=True, equalScaleZ=True)  # [debug]
    
    # * Top-level interface
    self.selectedNeuron = None  # the last selected SelectionNeuron, mainly for display and top-level output
    self.selectedTime = 0.0  # corresponding timestamp
    
    # * Allocate output images
    self.imageSalienceOut = np.zeros((self.retina.imageSize[1], self.retina.imageSize[0]), dtype=np.uint8) if self.context.options.gui else None  # salience neuron outputs
    self.imageSelectionOut = np.zeros((self.retina.imageSize[1], self.retina.imageSize[0]), dtype=np.uint8) if self.context.options.gui else None  # selection neuron outputs
  
  def process(self, imageIn, timeNow):
    keepRunning, imageRetinaOut = SimplifiedProjector.process(self, imageIn, timeNow)
    
    # * Update cortical layers (TODO move this to VisualCortex.update())
    # ** Salience neurons
    for salienceNeuron in self.salienceNeurons.neurons:
      salienceNeuron.update(timeNow)  # update every iteration
      #salienceNeuron.updateWithP(timeNow)  # update probabilistically
      #self.logger.debug("Salience neuron potential: {:.3f}, response: {:.3f}, I_e: {}, pixelValue: {}".format(salienceNeuron.potential, salienceNeuron.response, salienceNeuron.I_e, salienceNeuron.pixelValue))
      
    # ** Selection neurons (TODO mostly duplicated code, perhaps generalizable?)
    for selectionNeuron in self.selectionNeurons.neurons:
      selectionNeuron.update(timeNow)  # update every iteration
      #selectionNeuron.updateWithP(timeNow)  # update probabilistically
      #self.logger.debug("Selection neuron potential: {:.3f}, response: {:.3f}, pixelValue: {}".format(selectionNeuron.potential, selectionNeuron.response, selectionNeuron.pixelValue))
    
    # * Render output images and show them
    if self.context.options.gui:
      # ** Salience neurons
      self.imageSalienceOut.fill(0.0)
      for salienceNeuron in self.salienceNeurons.neurons:
        # Render salience neuron's receptive field with response-based pixel value (TODO cache int radii and pixel as tuple?)
        cv2.circle(self.imageSalienceOut, (salienceNeuron.pixel[0], salienceNeuron.pixel[1]), np.int_(salienceNeuron.rfRadius), 128)
        cv2.circle(self.imageSalienceOut, (salienceNeuron.pixel[0], salienceNeuron.pixel[1]), np.int_(salienceNeuron.rfCenterRadius), salienceNeuron.pixelValue, cv.CV_FILLED)
      cv2.imshow("Salience neurons", self.imageSalienceOut)
      
      # ** Selection neurons
      #numUninhibited = 0  # [debug]
      self.imageSelectionOut.fill(0.0)
      for selectionNeuron in self.selectionNeurons.neurons:
        # Render selection neuron's position with response-based pixel value (TODO build receptive field when synapses are made, or later, using a stimulus test phase?)
        #if selectionNeuron.pixelValue > 200: print "[{:.2f}] {}".format(timeNow, selectionNeuron)  # [debug]
        if not selectionNeuron.isInhibited:  # no point drawing black circles for inhibited neurons
          #numUninhibited += 1  # [debug]
          #cv2.circle(self.imageSelectionOut, (selectionNeuron.pixel[0], selectionNeuron.pixel[1]), self.imageSize[0] / 20, selectionNeuron.pixelValue, cv.CV_FILLED)  # only render the one selected neuron, later
          self.selectedNeuron = selectionNeuron
          self.selectedTime = timeNow
          self.selectedNeuron.inhibit(timeNow, neuron_inhibition_period + 0.75)  # inhibit selected neuron for a bit longer
          break  # first uninhibited SelectionNeuron will be our selected neuron
      #print "# Uninhibited selection neurons: {}".format(numUninhibited)  # [debug]
      cv2.circle(self.imageSelectionOut, (self.selectedNeuron.pixel[0], self.selectedNeuron.pixel[1]), self.imageSize[0] / 20, int(255 * exp(self.selectedTime - timeNow)), cv.CV_FILLED)  # draw selected neuron with a shade that fades with time (TODO more accurate receptive field?)
      cv2.imshow("Selection neurons", self.imageSelectionOut)  # actually, just selected neuron
      
      # ** Final output image
      self.imageOut = cv2.bitwise_and(self.retina.imageBGR, self.retina.imageBGR, mask=self.imageSelectionOut)
    
    return keepRunning, self.imageOut


if __name__ == "__main__":
  run(CorticalProjector, description="Process visual input through a (simplified) Retina and VisualCortex.")
