"""Functional models of neurons from the visual cortex."""

from math import exp, sqrt, hypot
import numpy as np

from ...neuron import Neuron, threshold_potential, action_potential_peak, action_potential_trough, refractory_period, Uniform


class SalienceNeuron(Neuron):
  """A spiking neuron that responds to center-surround contrast in its receptive field."""
  
  _str_attrs = ['id', 'pathway', 'pixel', 'response', 'potential', 'pixelValue']
  
  # Receptive field parameters
  #rf_radius_range = Uniform(low=0.025, high=0.2)  # fraction of visual field size [uniform]
  rf_radius_factor = Uniform(low=0.5, high=1.0)  # factor to multiply distance from center with, to get RF radius [radial]
  rf_radius_min = 8.0  # absolute min
  rf_radius_max_factor = 0.45  # relative to half-diagonal of retina (i.e. an approximation of retinal radius)
  rf_center_radius_range = Uniform(low=(1.0 / sqrt(2.0)) * 0.9, high=(1.0 / sqrt(2.0)) * 1.1)  # fraction of receptive field radius (NOTE in order to ensure equal center and surround area, center radius must be about 1/sqrt(2) of outer radius; otherwise use mean when computing center-surround difference)
  #rf_radius_range = Uniform(low=0.05, high=0.5)  # fraction of visual field size, largish, wide range [uniform]
  #rf_center_radius_range = Uniform(low=0.1, high=0.5)  # fraction of receptive field radius, smallish, wide range - not guaranteed to equalize center-surround areas
  
  # TODO Define types of salience neurons (feature-specific, and feature-agnostic) as tuple of feature neurons/maps to connect to
  
  # Electrophysiological parameters for Integrate-and-Fire method (model)
  R = 300.0e06  # Ohms; membrane resistance (~30-700Mohm)
  C = 3.0e-09  # Farads; membrane capacitance (~2-3nF)
  tau = R * C  # seconds; time constant (~100-1000ms)
  max_current = 200.0e-12  # Amperes; determines how input response affects potential
  
  # Miscellaneous parameters
  potential_scale = 255 / abs(action_potential_peak - Neuron.resting_potential.mu)  # factor used to convert cell potential to image pixel value
  
  def __init__(self, location, timeNow, system, pathway=None, imageSet=None, pixel=None):
    Neuron.__init__(self, location, timeNow)
    self.system = system
    self.pathway = pathway  # currently a string, i.e. pathway label (TODO keep reference to VisualPathway object with a base image member?)
    self.imageSet = imageSet
    self.pixel = pixel if pixel is not None else np.int_(location[:2])
    self.expDecayFactor = 0.0
    self.response = 0.0
    self.I_e = 0.0
    self.pixelValue = 0
    
    # Receptive field parameters
    #self.rfRadius = np.random.uniform(self.rf_radius_range.low, self.rf_radius_range.high) * self.system.imageSize[0]
    self.rfRadius = np.random.uniform(self.rf_radius_factor.low, self.rf_radius_factor.high) * \
        hypot(self.location[0] - self.system.imageCenter[0], self.location[1] - self.system.imageCenter[1])
    if self.rfRadius < self.rf_radius_min:
      self.rfRadius = self.rf_radius_min
    elif self.rfRadius > (self.rf_radius_max_factor * hypot(self.system.imageCenter[0], self.system.imageCenter[1])):
      self.rfRadius = self.rf_radius_max_factor * hypot(self.system.imageCenter[0], self.system.imageCenter[1])
    self.rfCenterRadius = np.random.uniform(self.rf_center_radius_range.low, self.rf_center_radius_range.high) * self.rfRadius
    
    # Define receptive field as two slice/index expressions, one for the whole and one for center
    # NOTE Since whole = center + surround => surround = whole - center
    #    and, response = center - surrond
    #   thus, response = center - (whole - center) = 2 * center - whole
    self.rfSlice = np.index_exp[
      max(self.pixel[1] - self.rfRadius, 0):min(self.pixel[1] + self.rfRadius, self.system.imageSize[1]),
      max(self.pixel[0] - self.rfRadius, 0):min(self.pixel[0] + self.rfRadius, self.system.imageSize[0])]
    self.rfCenterSlice = np.index_exp[
      max(self.pixel[1] - self.rfCenterRadius, 0):min(self.pixel[1] + self.rfCenterRadius, self.system.imageSize[1]),
      max(self.pixel[0] - self.rfCenterRadius, 0):min(self.pixel[0] + self.rfCenterRadius, self.system.imageSize[0])]
  
  def synapseWith(self, neuron, strength=None, gatekeeper=None):
    Neuron.synapseWith(self, neuron, strength, gatekeeper)
    # Pass along self reference for receptive field information to selection neuron
    if hasattr(neuron, 'inputNeurons'):
      neuron.inputNeurons.append(self)
  
  def updatePotential(self):
    # Gather response from feature map (TODO choose feature map based on type)
    rfWholeResponse = np.mean(self.imageSet[self.pathway][self.rfSlice])  # TODO check if mean() is appropriate here, or is sum() better (then we need to ensure +ve and -ve areas are roughly equal)?
    rfCenterResponse = np.mean(self.imageSet[self.pathway][self.rfCenterSlice])
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
    #print "[{:.2f}] {}, whole: {}, center: {}".format(self.timeCurrent, self, rfWholeResponse, rfCenterResponse)  # [debug]
    
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
  
  _str_attrs = ['id', 'pathway', 'pixel', 'potential', 'pixelValue']
  
  # Electrophysiological parameters for Integrate-and-Fire method (model)
  R = 300.0e06  # Ohms; membrane resistance (~30-700Mohm)
  C = 3.0e-09  # Farads; membrane capacitance (~2-3nF)
  tau = R * C  # seconds; time constant (~100-1000ms)
  
  # Receptive field parameters
  rf_radius_factor_default = 0.25
  
  # Miscellaneous parameters
  potential_scale = 255 / abs(action_potential_peak - Neuron.resting_potential.mu)  # factor used to convert cell potential to image pixel value
  
  def __init__(self, location, timeNow, system, pathway=None, pixel=None):
    Neuron.__init__(self, location, timeNow)
    self.system = system
    self.pathway = pathway
    self.pixel = pixel if pixel is not None else np.int_(location[:2])
    #self.expDecayFactor = 0.0
    self.pixelValue = 0
    self.inputNeurons = []  # will be populated when synapses are made by salience neurons
    self.rfRadius = self.rf_radius_factor_default * hypot(self.location[0] - self.system.imageCenter[0], self.location[1] - self.system.imageCenter[1])
  
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


class FeatureNeuron(Neuron):
  """A simple neuron with graded potential output and slow response meant to encode feature-based activation at a high level."""
  
  _str_attrs = ['id', 'pathway', 'pixel', 'potential', 'pixelValue']
  
  # Electrophysiological parameters for Integrate-and-Fire method (model)
  R = 500.0e06  # Ohms; membrane resistance (~30-700Mohm)
  C = 3.0e-09  # Farads; membrane capacitance (~2-3nF)
  tau = R * C  # seconds; time constant (~100-1000ms)
  
  # Miscellaneous parameters
  potential_scale = 255 / abs(action_potential_peak - Neuron.resting_potential.mu)  # factor used to convert cell potential to image pixel value
  
  def __init__(self, location, timeNow, system, pathway=None, pixel=None):
    Neuron.__init__(self, location, timeNow)
    self.system = system
    self.pathway = pathway
    self.pixel = pixel if pixel is not None else np.int_(location[:2])
    self.expDecayFactor = 0.0
    self.pixelValue = 0
  
  def updatePotential(self):
    # Decay potential
    self.expDecayFactor = exp(-self.deltaTime / self.tau)
    self.potential = self.resting_potential.mu + ((self.potentialLastUpdated - self.resting_potential.mu) * self.expDecayFactor)  # exponential decay: V(t) = V_r + ((V(t') - V_r) * (e ^ (-(t - t') / tau)))
    #self.potential -= (self.potentialLastUpdated - self.resting_potential.mu) * self.deltaTime / self.tau  # linear approximation of exponential decay: V(t) = V(t') - (V(t') - V_r) * ((t - t') / tau)
    # NOTE Linear approximation may cause underflow, but that shouldn't cause any problems
    
    # Accumulate/integrate incoming potentials
    self.potential += self.potentialAccumulated  # integrate signals accumulated from neighbors
    self.potentialAccumulated = 0.0  # reset accumulator (don't want to double count!)
    
    # Compute representative pixel value to render (TODO report potential here for logging and plotting)
    #print "[{:.2f}] {}".format(self.timeCurrent, self)  # [debug]
    self.pixelValue = int(np.clip(abs(self.potential - self.resting_potential.mu) * self.potential_scale, 0, 255))
    
    self.sendGradedPotential()
