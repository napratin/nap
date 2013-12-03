"""Photoreceptor neuron models."""

import numpy as np

from ..neuron import Neuron

class Photoreceptor(Neuron):
  """A specialized neuron that is sensitive to light."""
  
  _str_attrs = ['id', 'pixel', 'potential']
  
  def __init__(self, location, timeNow, retina=None, pixel=None):
    Neuron.__init__(self, location, timeNow)
    self.retina = retina
    self.pixel = pixel if pixel is not None else np.int_(location[:2])


class Rod(Photoreceptor):
  """A type of photoreceptor that is sensitive only to the intensity of light."""
  
  def updatePotential(self):
    Photoreceptor.updatePotential(self)


class Cone(Photoreceptor):
  """A type of photoreceptor that is sensitive to the frequency (color) of light."""
  
  def updatePotential(self):
    Photoreceptor.updatePotential(self)


if __name__ == "__main__":
  # Test
  p = Photoreceptor(np.float32([1.5, 2.4, 0.0]), 0.0)
  print p
  r = Rod(np.float32([1.2, -2.7, 0.2]), 0.1)
  r.updatePotential()
  print r
  c = Cone(np.float32([5.1, -1.2, 0.1]), 0.2)
  print c
