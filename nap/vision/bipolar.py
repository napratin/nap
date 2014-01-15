"""Bipolar cells."""

from math import exp
import numpy as np
from ..neuron import Neuron

class BipolarCell(Neuron):
  """A bipolar cell model."""
  
  # Bipolar cell types:-
  # * By response:-
  # ** Hyperpolarizing (H): Excitatory receptors, OFF-center
  # ** Depolarizing (D): Inhibitory receptors, ON-center
  # * By connectivity/morphological properties:-
  # ** Midget (MB): Single cone input, mostly found in fovea
  # *** Flat Midget (FMB)
  # *** Invaginating Midget (IMB)
  # ** Diffuse (DB): Cones only, 6-9 subtypes (DB1, DB2 ...) with axons terminating at gradually outer layers
  # ** Rod (RB): Rods only, ON-center only, axons terminating in outermost layer
  # ** Blue-cone (BB): Blue cone only
  # Giant Bistratified (GBB): Axons terminating in two distinct layers - one inner, one outer
  
  # Electrophysiological parameters for Integrate-and-Fire method (model)
  R = 300.0e06  # Ohms; membrane resistance (~30-700Mohm)
  C = 3.0e-09  # Farads; membrane capacitance (~2-3nF)
  tau = R * C  # seconds; time constant (~100-1000ms)
  
  # Miscellaneous parameters
  potential_scale = 255 / abs(Neuron.resting_potential.mu / 2)  # factor used to convert cell potential to image pixel value
  
  def __init__(self, location, timeNow, retina):
    Neuron.__init__(self, location, timeNow)
    self.retina = retina
    self.pixel = np.int_(location[:2])
    self.expDecayFactor = 0.0
    self.pixelValue = 0
  
  def updatePotential(self):
    # NOTE: Biploar cells use graded potentials
    # Differential equation solution, decay only (Method 4, similar to Photoreceptor)
    self.expDecayFactor = exp(-self.deltaTime / self.tau)
    self.potential = self.resting_potential.mu + ((self.potentialLastUpdated - self.resting_potential.mu) * self.expDecayFactor)  # V_m = V_r + (V(t_0) * (e ^ (-(t - t_0) / tau)))
    
    # Accumulate/integrate incoming potentials
    self.potential += self.potentialAccumulated  # integrate signals accumulated from neighbors
    self.potentialAccumulated = 0.0  # reset accumulator (don't want to double count!)
    
    # Compute a value to render
    self.pixelValue = int(np.clip(abs(self.potential - self.resting_potential.mu) * self.potential_scale, 0, 255))
    
    #self.sendGradedPotential()
