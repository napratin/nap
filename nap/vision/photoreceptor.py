"""Photoreceptor neuron models."""

from math import exp
import random
import numpy as np

from ..neuron import Neuron, action_potential_trough, action_potential_peak, Normal

class Photoreceptor(Neuron):
  """A specialized neuron that is sensitive to light."""
  
  _str_attrs = ['id', 'pixel', 'potential']
  
  # Electrophysiological parameters for Integrate-and-Fire phototransduction method (model)
  R = 300.0e06  # Ohms; membrane resistance (~30-700Mohm)
  C = 3.0e-09  # Farads; membrane capacitance (~2-3nF)
  tau = R * C  # seconds; time constant (~100-1000ms)
  dark_current = 100.0e-12  # Amps; convert normalized potential (0..1) to external current I_e (~100-200pA)
  dark_potential = Neuron.resting_potential.mu + R * dark_current  # -0.04 mV
  # NOTE See: Kawai et. al. 2001, Na+ Action Potentials in Human Photoreceptors, Neuron 30-2, pp. 451-458
  
  '''
  # Electrophysiological parameters for other methods (models)
  potential_decay = 1.0  # per-sec.; rate at which potential decays trying to reach equillibrium
  response_to_potential = -0.07  # normalized response (0..1) to absolute potential (-0.04..-0.07)
  response_resistance = 1.0
  response_to_delta_potential = -0.25  # multiplication factor to convert normalized response value (0..1) to delta cell potential (approx. range: 0.01..0.1; negative because photoreceptors hyperpolarize in response to stimulus) [TODO use non-linear relationship based on current potential to avoid overshoot (and perhaps deltaTime as well?)]
  potential_range = np.float32([action_potential_trough.mu - 3*action_potential_trough.sigma, action_potential_peak])  # [deprecated: photoreceptors don't obey this range]
  '''
  
  # Miscellaneous parameters
  potential_scale = 1.0 / abs(Neuron.resting_potential.mu - dark_potential)  # factor used to convert cell potential to image pixel value
  
  def __init__(self, location, timeNow, retina, pixel=None):
    Neuron.__init__(self, location, timeNow)
    self.retina = retina
    self.pixel = pixel if pixel is not None else np.int_(location[:2])
    self.expDecayFactor = 0.0
    self.I_e = 0.0
  
  def updatePotential(self):
    '''
    Phototransduction (NOTE: Photoreceptors use graded potentials; no action potentials)
    
    Neuron model:
    V_m: membrane potential, V_r: resting potential
    R: membrane resistance, C: membrane capacitance
    I_e: external (input) current
    
    I_R = (V_m - V_r) / R  # resistive (leakage) current across membrane as resistor
    I_C = C * (dV_m / dt)  # capacitive current across membrane as capacitor
    I_R + I_C = I_e  # Kirschoff's law of conservation of current [NOTE I_e direction]
    
    Therefore,
    I_C = I_e - I_R
    C * (dV_m / dt) = I_e - ((V_m - V_r) / R)
    
    Solving for V_m,
    V_m = V_r + V(t_0) * (e ^ (-(t - t_0) / tau))           ...when I_e = 0 (discharging)
    V_m = V_r + R * I_e * (1 - e ^ (-(t - t_0) / tau))      ...when I_e is non-zero (charging)
      where t_0 = last time, t = current time, V(t_0) = voltage at time t_0 (relative to V_r), tau = R * C, and t is time elapsed since time 0
    
    Combining these two states to model simultaneous charging and discharging,
    V_m = V_r + (V(t_0) * (e ^ (-(t - t_0) / tau))) + (R * I_e * (1 - e ^ (-(t - t_0) / tau)))
    '''
    
    '''
    # * Method 1: Decay, accumulate potential using regular technique
    # ** Accumulate/integrate potential change due to response current
    self.accumulate(self.response_to_delta_potential * self.response * self.deltaTime)  # TODO don't accumulate (potential values are going off the charts), use graded potential instead to directly reach appropriate potential level (with some delay, and dependent on current potential value)?
    
    # ** Decay potential
    self.potential -= self.potential_decay * (self.potential - self.resting_potential.mu) * self.deltaTime  # approximated exponential decay
    
    # ** Add accumulated potential
    self.potential += self.potentialAccumulated  # integrate signals accumulated from neighbors
    self.potentialAccumulated = 0.0  # reset accumulator (don't want to double count!)
    '''
    
    '''
    # * Method 2: Two-step - decay, then response - using simplified technique
    self.potential -= self.potential_decay * (self.potential - self.resting_potential.mu) * self.deltaTime
    self.potential += self.response_resistance * (self.response_to_potential * self.response) * self.deltaTime
    '''
    
    '''
    # * Method 3: One-step response and decay (works fine)
    #self.potential += ((self.response_to_potential * self.response) - self.potential_decay * (self.potential - self.resting_potential.mu)) * self.deltaTime
    '''
    
    # * Method 4: Differential equation solution
    self.expDecayFactor = exp(-self.deltaTime / self.tau)
    self.I_e = self.dark_current * (1.0 - self.response)  # I_e = dark_current when response == 0
    self.potential = self.resting_potential.mu + ((self.potentialLastUpdated - self.resting_potential.mu) * self.expDecayFactor) + (self.R * self.I_e * (1.0 - self.expDecayFactor))  # V_m = V_r + (V(t_0) * (e ^ (-(t - t_0) / tau))) + (R * I_e * (1 - e ^ (-(t - t_0) / tau)))
    
    '''
    # * Method 5: Weighted response effect on potential
    target_potential = self.resting_potential.mu - self.R * self.response
    self.potential = (self.potentialLastUpdated + target_potential) / 2
    '''


# TODO Check these values to establish correct mapping between light frequencies (in nm) and color hues (0..180)
max_hue = 180.0  # degrees; min is implicitly 0
min_hue_freq = 390.0
max_hue_freq = 600.0
hue_freq_range = max_hue_freq - min_hue_freq
freq_to_hue_factor = max_hue / hue_freq_range
def freqToHue(freq, rel=False):
  """Convert frequency (in nm) to hue angle (0..180)."""
  # TODO Move to a common utility module? Also, clean-up, improve and validate this.
  if rel:
    return freq * freq_to_hue_factor
  return ((max_hue_freq - freq) * freq_to_hue_factor) % max_hue  # clip or modulo?


class PhotoreceptorType:
  def __init__(self, name, freqResponse, hueSensitivity, satSensitivity, valSensitivity, occurrence):
    # TODO Add distribution parameter(s)
    self.name = name
    self.freqResponse = freqResponse
    self.hueResponse = Normal(mu=freqToHue(self.freqResponse.mu), sigma=freqToHue(self.freqResponse.sigma, rel=True))
    self.hue = int(self.hueResponse.mu)  # frequently used (subtracted from uint8 image)
    self.hueSensitivity = hueSensitivity
    self.satSensitivity = satSensitivity
    self.valSensitivity = valSensitivity
    # TODO Need non-linear response to hue, sat, value
    self.hueResponseFactor = self.hueSensitivity / 180.0
    self.satResponseFactor = self.satSensitivity / 255.0
    self.valResponseFactor = self.valSensitivity / 255.0
    self.responseFactor = self.hueResponseFactor * self.satResponseFactor * self.valResponseFactor  # used to normalize response value to [0, 1] range
    self.occurrence = occurrence
    #print self  # [debug]
  
  def __str__(self):
    return "PhotoreceptorType: {{ name: {}, freqResponse: {}, hueResponse: {}, hueSensitivity: {}, occurrence: {} }}".format(self.name, self.freqResponse, self.hueResponse, self.hueSensitivity, self.occurrence)


class Rod(Photoreceptor):
  """A type of photoreceptor that is sensitive mainly to the intensity of light."""
  
  # NOTE Rod cells are sensitive to a particular frequency range (peak around 498nm)
  rod_type = PhotoreceptorType('Rod', Normal(mu=498.0, sigma=50.0), 1.0, 1.0, 1.0, 1.0)
  
  def __init__(self, location, timeNow, retina, pixel=None, coneType=None):
    Photoreceptor.__init__(self, location, timeNow, retina, pixel)
    self.response = 0.0
    self.pixelValue = 0  # output pixel value
  
  def updatePotential(self):
    # Phototransduction: Accumulate some potential based on light's intensity (value) in retina
    self.response = self.retina.imageRod[self.pixel[1], self.pixel[0]]  # response value computed by retina for each photoreceptor
    Photoreceptor.updatePotential(self)
    self.pixelValue = int(np.clip(abs(self.potential - self.dark_potential) * self.potential_scale * self.retina.imageHSV[self.pixel[1], self.pixel[0]][2], 0, 255))  # deviation from dark potential
    #self.pixelValue = int(np.clip((self.potential - self.potential_range[0]) * self.potential_scale * self.retina.imageHSV[self.pixel[1], self.pixel[0]][2], 0, 255))  # potential compared to absolute range
    
    self.sendGradedPotential()


class Cone(Photoreceptor):
  """A type of photoreceptor that is sensitive to the frequency (color) of light."""
  
  # NOTE The ratio of different types of cone cells can vary a lot from individual to individual, here we pick a representative distribution; peak response frequencies are also somewhat debatable
  cone_types = [ PhotoreceptorType('S', Normal(mu=440.0, sigma=20.0), 3.0, 1.0, 1.0, 0.04), PhotoreceptorType('M', Normal(mu=530.0, sigma=25.0), 2.5, 1.0, 1.0, 0.32), PhotoreceptorType('L', Normal(mu=570.0, sigma=30.0), 2.0, 1.0, 1.0, 0.64) ]  # S = blue, M = green, L = red [TODO check values, esp. sensitivity and occurrence]
  cone_probabilities = np.float32([cone_type.occurrence for cone_type in cone_types])  # occurrence probabilities [TODO normalize so that they sum to 1?]
  
  def __init__(self, location, timeNow, retina=None, pixel=None, coneType=None):
    Photoreceptor.__init__(self, location, timeNow, retina, pixel)
    # Make cones sensitive to a certain type
    self.coneType = coneType if ((coneType is not None) and (coneType in self.cone_types)) else np.random.choice(self.cone_types, p=self.cone_probabilities)
    #self.hue = np.random.normal(self.coneType.hueResponse.mu, self.coneType.hueResponse.sigma) % max_hue  # [deprecated: use self.coneType.hueResponse directly in updatePotential]
    self.freq = np.random.normal(self.coneType.freqResponse.mu, self.coneType.freqResponse.sigma)  # [deprecated, not useful other than for plotting]
    #print self.freq  # [debug]
    self.hue = freqToHue(self.freq)
    self.pixelValue = np.uint8([0, 0, 0])
  
  def updatePotential(self):
    # Phototransduction: Accumulate some potential based on light's color (hue), saturation and value in retina
    self.response = self.retina.imagesCone[self.coneType.name][self.pixel[1], self.pixel[0]]  # response value computed by retina for each photoreceptor [TODO include sensitivity term here?]
    Photoreceptor.updatePotential(self)
    self.pixelValue = np.uint8(np.clip(abs(self.potential - self.dark_potential) * self.potential_scale * self.retina.imageBGR[self.pixel[1], self.pixel[0]], 0, 255))  # deviation from dark potential
    #self.pixelValue = np.uint8(np.clip((self.potential - self.potential_range[0]) * self.potential_scale * self.retina.imageHSV[self.pixel[1], self.pixel[0]], 0, 255))  # potential compared to absolute range
    
    self.sendGradedPotential()


if __name__ == "__main__":
  # Test
  p = Photoreceptor(np.float32([1.5, 2.4, 0.0]), 0.0)
  print p
  r = Rod(np.float32([1.2, -2.7, 0.2]), 0.1)
  r.updatePotential()
  print r
  c = Cone(np.float32([5.1, -1.2, 0.1]), 0.2)
  print c
