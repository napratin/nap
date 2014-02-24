"""Bipolar cells."""

from math import exp
from collections import namedtuple
import numpy as np

from ..neuron import Neuron

# Simple structure to define ON- and OFF- Bipolar cell types (polarity = -1 means depolarizing/inhibitory/ON, and vice-versa)
BipolarType = namedtuple('BipolarType', ['name', 'polarity', 'occurrence'])


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
  
  # Simplified set of Bipolar types and their occurrence probabilities
  bipolar_types = [ BipolarType('ON', -1.0, 0.5), BipolarType('OFF', 1.0, 0.5) ]
  bipolar_probabilities = np.float32([ bipolar_type.occurrence for bipolar_type in bipolar_types ])
  
  # Electrophysiological parameters for Integrate-and-Fire method (model)
  R = 300.0e06  # Ohms; membrane resistance (~30-700Mohm)
  C = 3.0e-09  # Farads; membrane capacitance (~2-3nF)
  tau = R * C  # seconds; time constant (~100-1000ms)
  
  # Miscellaneous parameters
  potential_scale = 255 / abs(Neuron.resting_potential.mu / 2)  # factor used to convert cell potential to image pixel value
  
  def __init__(self, location, timeNow, retina, bipolarType=None):
    Neuron.__init__(self, location, timeNow)
    self.retina = retina
    self.pixel = np.int_(location[:2])
    #self.bipolarType = bipolarType if ((bipolarType is not None) and (bipolarType in self.bipolar_types)) else np.random.choice(self.bipolar_types, p=self.bipolar_probabilities)  # TODO debug this
    self.bipolarType = self.bipolar_types[0]
    # TODO Ensure Rod Bipolars are ON-center only?
    #print "Bipolar type:", self.bipolarType
    self.expDecayFactor = 0.0
    self.pixelValue = 0
  
  def updatePotential(self):
    # NOTE: Bipolar cells use graded potentials
    # Differential equation solution, decay only (similar to Photoreceptor, Method 4)
    self.expDecayFactor = exp(-self.deltaTime / self.tau)
    self.potential = self.resting_potential.mu + ((self.potentialLastUpdated - self.resting_potential.mu) * self.expDecayFactor)  # V(t) = V_r + ((V(t') - V_r) * (e ^ (-(t - t') / tau)))
    
    # Accumulate/integrate incoming potentials (TODO hyperpolarize/depolarize based on polarity)
    self.potential += self.bipolarType.polarity * self.potentialAccumulated  # integrate signals accumulated from neighbors
    self.potentialAccumulated = 0.0  # reset accumulator (don't want to double count!)
    
    # Compute a value to render
    self.pixelValue = int(np.clip(abs(self.potential - self.resting_potential.mu) * self.potential_scale, 0, 255))
    
    #self.sendGradedPotential()
