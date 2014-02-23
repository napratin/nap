"""Simplified model of the visual cortex with appropriate neuron types."""

from math import exp
import numpy as np
import cv2
import cv2.cv as cv

from lumos.input import run

from ...neuron import Neuron, NeuronGroup, threshold_potential, action_potential_peak, action_potential_trough, refractory_period, Uniform, MultivariateUniform, MultivariateNormal, plotNeuronGroups
from .retina import Retina, SimplifiedProjector

class SalienceNeuron(Neuron):
  """A spiking neuron that responds to center-surround contrast in its receptive field."""
  
  # Receptive field parameters
  rf_radius_range = Uniform(low=0.025, high=0.2)  # fraction of visual field size
  rf_center_radius_range = Uniform(low=0.4, high=0.6)  # fraction of receptive field radius
  
  # TODO Define types of salience neurons (feature-specific, and feature-agnostic) as tuple of feature neurons/maps to connect to
  
  # Electrophysiological parameters for Integrate-and-Fire method (model)
  R = 300.0e06  # Ohms; membrane resistance (~30-700Mohm)
  C = 3.0e-09  # Farads; membrane capacitance (~2-3nF)
  tau = R * C  # seconds; time constant (~100-1000ms)
  max_current = 100.0e-12
  
  # Miscellaneous parameters
  potential_scale = 255 / abs(Neuron.resting_potential.mu / 2)  # factor used to convert cell potential to image pixel value
  
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
    # Differential equation solution, decay only
    self.expDecayFactor = exp(-self.deltaTime / self.tau)
    self.potential = self.resting_potential.mu + ((self.potentialLastUpdated - self.resting_potential.mu) * self.expDecayFactor)  # V_m = V_r + (V(t_0) * (e ^ (-(t - t_0) / tau)))
    
    # Ensure we are not in refractory period (TODO generalize to any form of inhibition?)
    if (self.timeCurrent - self.timeLastFired) >= refractory_period:
      # Accumulate/integrate incoming potentials from feature map (TODO choose feature map based on type)
      rfWholeResponse = np.mean(self.retina.imageSalience[self.rfSlice])  # TODO check if mean() is appropriate here?
      rfCenterResponse = np.mean(self.retina.imageSalience[self.rfCenterSlice])
      self.response = np.clip(2 * rfCenterResponse - rfWholeResponse, 0.0, 1.0)  # response = 2 * center - whole
      self.I_e = self.max_current * self.response
      self.potential += (self.R * self.I_e * (1.0 - self.expDecayFactor))  # V_m += (R * I_e * (1 - e ^ (-(t - t_0) / tau)))
      
      # Compute a value to render
      self.pixelValue = int(np.clip(abs(self.potential - self.resting_potential.mu) * self.potential_scale, 0, 255))
      
      # Check for action potential event
      if self.potential > threshold_potential:
        self.actionPotential()
      
      # TODO Report potential here for logging and plotting
      
      # Fire action potential, if we've reached peak
      if self.potential >= action_potential_peak:
        self.fireActionPotential()
        self.timeLastFired = self.timeCurrent
        self.potential = np.random.normal(action_potential_trough.mu, action_potential_trough.sigma)  # repolarization/falling phase (instantaneous)


# TODO Define VisualCortex class that would contain the different cortical layers and expose an interface to the visual system (with retina passed in through context? or context.systems/context.components?)


class CorticalProjector(SimplifiedProjector):
  """A SimplifiedProjector-based type that runs visual input through a simplified Retina as well as higher cortical layers."""
  
  num_salience_neurons = 400
  
  def __init__(self, retina=None):
    SimplifiedProjector.__init__(self, retina if retina is not None else Retina())
    
    # Create layer of SalienceNeurons (TODO move this to VisualCortex, introduce magno and parvo types)
    #self.salienceNeuronDistribution = MultivariateNormal(mu=self.retina.center, cov=(np.float32([self.retina.center[0] ** 2, self.retina.center[0] ** 2, 1.0]) * np.identity(3, dtype=np.float32)))
    self.salienceNeuronDistribution = MultivariateUniform(lows=[0.0, 0.0, 0.0], highs=[self.retina.imageSize[1], self.retina.imageSize[0], 0.0])
    self.salienceNeurons = NeuronGroup(numNeurons=self.num_salience_neurons, timeNow=0.0, neuronTypes=[SalienceNeuron], bounds=self.retina.bounds, distribution=self.salienceNeuronDistribution, retina=self.retina)  # TODO use timeNow when moved to VisualCortex
    self.salienceNeuronPlotColor = 'coral'
    
    #plotNeuronGroups([self.salienceNeurons], groupColors=[self.salienceNeuronPlotColor], showConnections=True, equalScaleZ=True)  # [debug]
    
    self.imageSalienceOut = np.zeros((self.retina.imageSize[1], self.retina.imageSize[0]), dtype=np.uint8) if self.context.options.gui else None  # image to render salience neuron outputs
  
  def process(self, imageIn, timeNow):
    keepRunning, imageRetinaOut = SimplifiedProjector.process(self, imageIn, timeNow)
    
    # Update cortical layers (TODO move this to VisualCortex.update())
    # * Salience neurons
    if self.context.options.gui: self.imageSalienceOut.fill(0.0)
    for salienceNeuron in self.salienceNeurons.neurons:
      salienceNeuron.update(timeNow)  # update every iteration
      #salienceNeuron.updateWithP(timeNow)  # update probabilistically
      #self.logger.debug("Salience neuron potential: {:.3f}, response: {:.3f}, I_e: {}, pixelValue: {}".format(salienceNeuron.potential, salienceNeuron.response, salienceNeuron.I_e, salienceNeuron.pixelValue))
      if self.context.options.gui:
        # Render salience neuron's receptive field with response-based pixel value (TODO cache int radii and pixel as tuple?)
        cv2.circle(self.imageSalienceOut, (salienceNeuron.pixel[0], salienceNeuron.pixel[1]), np.int_(salienceNeuron.rfRadius), 128)
        cv2.circle(self.imageSalienceOut, (salienceNeuron.pixel[0], salienceNeuron.pixel[1]), np.int_(salienceNeuron.rfCenterRadius), salienceNeuron.pixelValue, cv.CV_FILLED)
      
      # * Selection neurons
    
    return keepRunning, self.imageSalienceOut


if __name__ == "__main__":
  run(CorticalProjector, description="Process visual input through a (simplified) Retina and VisualCortex.")
