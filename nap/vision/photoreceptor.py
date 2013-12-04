"""Photoreceptor neuron models."""

import random
import numpy as np

from ..neuron import Neuron, action_potential_trough, action_potential_peak, Normal

class Photoreceptor(Neuron):
  """A specialized neuron that is sensitive to light."""
  
  _str_attrs = ['id', 'pixel', 'potential']
  
  potential_range = np.float32([action_potential_trough.mu - 3*action_potential_trough.sigma, action_potential_peak])
  potential_scale = 1.0 / (potential_range[1] - potential_range[0])  # factor used to convert cell potential to image pixel value
  
  def __init__(self, location, timeNow, retina=None, pixel=None):
    Neuron.__init__(self, location, timeNow)
    self.retina = retina
    self.pixel = pixel if pixel is not None else np.int_(location[:2])


class Rod(Photoreceptor):
  """A type of photoreceptor that is sensitive only to the intensity of light."""
  
  value_to_delta_potential = 1.0 / (8 * 255)  # multiplication factor to convert image pixel value (0..255) to delta cell potential (approx. range: 0.01..0.1)
  
  def __init__(self, location, timeNow, retina=None, pixel=None, coneType=None):
    Photoreceptor.__init__(self, location, timeNow, retina, pixel)
    # TODO Make rods sensitive to higher/lower intensity?
    self.pixelValue = 0
  
  def updatePotential(self):
    # Phototransduction: Accumulate some potential based on light's intensity (value) in retina
    self.accumulate(self.value_to_delta_potential * self.retina.imageHSV[self.pixel[1], self.pixel[0]][2])
    Photoreceptor.updatePotential(self)
    self.pixelValue = int(np.clip((self.potential - self.potential_range[0]) * self.potential_scale * self.retina.imageHSV[self.pixel[1], self.pixel[0]][2], 0, 255))


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


class ConeType:
  def __init__(self, name, freqResponse, sensitivity, occurrence):
    # TODO Add cone distribution parameter(s)
    self.name = name
    self.freqResponse = freqResponse
    self.hueResponse = Normal(mu=freqToHue(self.freqResponse.mu), sigma=freqToHue(self.freqResponse.sigma, rel=True))
    self.sensitivity = sensitivity
    self.occurrence = occurrence
    #print self  # [debug]
  
  def __str__(self):
    return "ConeType: {{ name: {}, freqResponse: {}, hueResponse: {}, sensitivity: {}, occurrence: {} }}".format(self.name, self.freqResponse, self.hueResponse, self.sensitivity, self.occurrence)


class Cone(Photoreceptor):
  """A type of photoreceptor that is sensitive to the frequency (color) of light."""
  
  cone_types = [ ConeType('S', Normal(mu=440.0, sigma=20.0), 0.9, 0.04), ConeType('M', Normal(mu=540.0, sigma=25.0), 0.5, 0.32), ConeType('L', Normal(mu=570.0, sigma=30.0), 0.5, 0.64) ]  # S = blue, M = green, L = red [TODO check values, esp. sensitivity and occurrence]
  cone_probabilities = np.float32([cone_type.occurrence for cone_type in cone_types])  # occurrence probabilities [TODO normalize so that they sum to 1?]
  
  #cone_hues = { 'L': 20, 'M': 50, 'S': 120 }  # S = blue, M = green, L = red [deprecated]
  #cone_hue_noise = 10.0  # standard deviation of hue sensitivity distribution around L, M, S centers [deprecated]
  value_to_delta_potential = 1.0 / (180 * 255 * 255)  # multiplication factor to convert (combined) image pixel value (0..180*255*255) to delta cell potential (approx. range: 0.01..0.1) [TODO can be replaced by ConeType.sensitivity]
  
  def __init__(self, location, timeNow, retina=None, pixel=None, coneType=None):
    Photoreceptor.__init__(self, location, timeNow, retina, pixel)
    # Make cones sensitive to a certain type
    #self.coneType = coneType if ((coneType is not None) and (coneType in self.cone_hues)) else random.choice(self.cone_hues.keys())
    #self.hue = np.clip(self.cone_hues[self.coneType] + np.random.normal(0.0, self.cone_hue_noise), 0, 180)  # [deprecated]
    self.coneType = coneType if ((coneType is not None) and (coneType in self.cone_types)) else np.random.choice(self.cone_types, p=self.cone_probabilities)
    #self.hue = np.random.normal(self.coneType.hueResponse.mu, self.coneType.hueResponse.sigma) % max_hue  # [deprecated: use self.coneType.hueResponse directly in updatePotential]
    self.freq = np.random.normal(self.coneType.freqResponse.mu, self.coneType.freqResponse.sigma)  # [deprecated, not useful other than for plotting]
    #print self.freq  # [debug]
    self.hue = freqToHue(self.freq)
    self.pixelValue = np.uint8([0, 0, 0])
  
  def updatePotential(self):
    # Phototransduction: Accumulate some potential based on light's color (hue), saturation and value in retina
    hsv = self.retina.imageHSV[self.pixel[1], self.pixel[0]]
    self.accumulate(self.value_to_delta_potential * (180 - (abs(hsv[0] - self.hue) % 180)) * hsv[1] * hsv[2])  # TODO use self.coneType.hueResponse and .sensitivity to get more realistic response
    Photoreceptor.updatePotential(self)
    self.pixelValue = np.uint8(np.clip((self.potential - self.potential_range[0]) * self.potential_scale * self.retina.imageBGR[self.pixel[1], self.pixel[0]], 0, 255))


if __name__ == "__main__":
  # Test
  p = Photoreceptor(np.float32([1.5, 2.4, 0.0]), 0.0)
  print p
  r = Rod(np.float32([1.2, -2.7, 0.2]), 0.1)
  r.updatePotential()
  print r
  c = Cone(np.float32([5.1, -1.2, 0.1]), 0.2)
  print c
