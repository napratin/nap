"""A neuron model based on a biologically inspired neuronal architecture."""

import logging
from time import time
from math import exp, pi
from collections import namedtuple, deque
import numpy as np
import cv2

from matplotlib.pyplot import figure, plot, axis, show, subplots_adjust
from mpl_toolkits.mplot3d import Axes3D

# Different neuron distributions
Normal = namedtuple('Normal', ['mu', 'sigma'])  # a normal distribution, defined by mean (mu) and std. dev. (sigma)
MultivariateNormal = namedtuple('MultivariateNormal', ['mu', 'cov'])  # a multivariate normal distribution, defined by mean vector (mu) of length N and covariance matrix (cov) of size NxN
SymmetricNormal = namedtuple('SymmetricNormal', Normal._fields + ('center',))
SymmetricLogNormal = namedtuple('SymmetricLogNormal', SymmetricNormal._fields)

# Neuron membrane potential levels
resting_potential = Normal(-0.07, 0.001)  # volts (mean, s.d.); resting / equillibrium potential
threshold_potential = -0.055  # volts; level that triggers action potential
action_potential_peak = 0.04  # volts; maximum potential reached during an action potential event
action_potential_trough = Normal(-0.08, 0.001)  # volts; minimum potential reached due to hyperpolarization during an action potential event

# Synaptic strength distribution (TODO find out from literature what these values should be)
synaptic_strength = Normal(0.011, 0.001)  # essentially volts (mean, s.d.); potential transmitted to post-synaptic neuron when the pre-synaptic neuron undergoes an action potential event

# Timing, decay and dynamic action potential parameters
potential_decay = 1.0  # per-sec.; rate at which potential decays trying to reach equillibrium
self_depolarization_rate = 0.75  # volts per sec.; rate at which potential rises during an action potential event
refractory_period = 0.1  # secs.; minimum time between two action potentials (should be an emergent effect when simulating action potential in detail)
min_update_time = 0.025  # secs.; minimum time between updates
synapse_inhibition_period = 0.5  # secs.; duration for which the effect of inhibition lasts at a synapse

# Graph parameters
plot_colors = 'bgrcmy'


class Projection:
  pass


class Dendrite(Projection):
  pass


class Axon(Projection):
  pass


class GrowthCone:
  default_maxLength = 15.0  # maximum length a projection with these growth cone parameters can grow to
  default_spreadFactor = 0.5  # radius at length L = spreadFactor * L
  
  def __init__(self, direction, maxLength=default_maxLength, spreadFactor=default_spreadFactor):
    self.direction = direction  # unit vector
    self.maxLength = maxLength
    self.spreadFactor = spreadFactor
  
  def score(self, anchor, target):
    """Return the probability that this growth cone starting at anchor would reach target."""
    dist_vec = target - anchor
    projection = np.dot(self.direction, dist_vec)
    if projection < 0.0 or projection > self.maxLength:
      return 0.0
    else:
      perp_distance = np.sqrt(np.linalg.norm(dist_vec, ord=2)**2 - projection**2)
      perp_limit = projection * self.spreadFactor
      return (perp_limit - perp_distance) / perp_limit if perp_distance < perp_limit else 0.0
  
  def __str__(self):
    return "GrowthCone: {{ direction: {}, maxLength: {}, spreadFactor: {} }}".format(self.direction, self.maxLength, self.spreadFactor)


class Synapse:
  """A synapse with references to pre-synaptic and post-synaptic neurons, and an optional gatekeeper neuron."""
  def __init__(self, pre, post, strength=None, gatekeeper=None):
    self.pre = pre
    self.post = post
    self.strength = strength if strength is not None else np.random.normal(synaptic_strength.mu, synaptic_strength.sigma)
    self.gatekeeper = gatekeeper  # NOTE do we need to store this here?
    #print "Synapse.__init__(): pre = {}, post = {}, strength = {}, gatekeeper = {}".format(self.pre.id, self.post.id, self.strength, self.gatekeeper.id if self.gatekeeper is not None else None)
    # TODO implement neuron-level inhibition (?), and learning/boosting of synaptic strength
    
    self.isInhibited = False
    self.uninhibitAt = -1.0
  
  def transmitActionPotential(self, timeNow):
    if self.isInhibited and self.uninhibitAt <= timeNow:
      self.isInhibited = False
    #print "Synapse.transmitActionPotential(): timeNow = {}, self.uninhibitAt = {} [{}]".format(timeNow, self.uninhibitAt, "INHIBITED" if self.isInhibited else "UNINHIBITED")
    
    if not self.isInhibited:
      self.post.accumulate(self.strength)
  
  def inhibit(self, timeNow, duration=synapse_inhibition_period):
    #print "Synapse.inhibit(): timeNow = {}, duration = {}".format(timeNow, duration)
    self.isInhibited = True
    self.uninhibitAt = timeNow + duration


class Neuron:
  """A simple excitable neuron cell with synaptic connections, potential accumulation and decay functionality."""
  
  id_ctr = 0  # auto-incremented counter to assign unique IDs to instances
  _str_attrs = ['id', 'location', 'potential']  # which attributes to include in string representation; subclasses can override this
  
  def __init__(self, location, timeNow):
    self.id = Neuron.id_ctr
    Neuron.id_ctr += 1
    self.location = location  # location in 3-space
    self.timeLastFired = self.timeLastUpdated = self.timeCurrent = timeNow
    self.deltaTime = 0.0
    
    self.potential = np.random.normal(resting_potential.mu, resting_potential.sigma)  # current membrane potential
    self.potentialAccumulated = 0.0  # potential accumulated from synaptic inputs
    
    self.synapses = list()
    self.gatedSynapses = list()
    
    self.timeLastPlotted = self.timeCurrent  # [graph]
    self.potentialLastPlotted = self.potential  # [graph]
    self.plotColor = plot_colors[self.id % len(plot_colors)]  # [graph]
  
  def synapseWith(self, neuron, strength=None, gatekeeper=None):
    s = Synapse(self, neuron, strength, gatekeeper)
    self.synapses.append(s)
    if gatekeeper is not None:
      gatekeeper.gateSynapse(s)
  
  def gateSynapse(self, synapse):
    self.gatedSynapses.append(synapse)
  
  def accumulate(self, deltaPotential):
    self.potentialAccumulated += deltaPotential
  
  def update(self, timeNow):
    self.timeCurrent = timeNow
    self.deltaTime = self.timeCurrent - self.timeLastUpdated
    if self.deltaTime < min_update_time:
      return
    
    self.updatePotential()
    self.timeLastUpdated = self.timeCurrent
  
  def updatePotential(self):
    # Fire action potential, if we've reached peak
    if self.potential >= action_potential_peak:
      self.fireActionPotential()
      self.timeLastFired = self.timeCurrent
      self.potential = np.random.normal(action_potential_trough.mu, action_potential_trough.sigma)  # repolarization/falling phase (instantaneous)
    
    # Decay potential
    #self.potential = resting_potential.mu + (self.potential - resting_potential.mu) * exp(-potential_decay * self.deltaTime)  # exponential decay
    self.potential -= potential_decay * (self.potential - resting_potential.mu) * self.deltaTime  # approximated exponential decay
    
    # Accumulate/integrate incoming potentials
    self.potential += self.potentialAccumulated  # integrate signals accumulated from neighbors
    self.potentialAccumulated = 0.0  # reset accumulator (don't want to double count!)
    
    # Check for action potential event
    if self.potential > threshold_potential and (self.timeCurrent - self.timeLastFired) >= refractory_period:
      self.actionPotential()
    
    #print self.id, self.timeCurrent, self.potential  # [log: potential]
    # TODO This is the ideal point to gather potential observation; "Fire action potential" step should come immediately after this (instead of at the beginning of updatePotential) in order to prevent any posible delays
    # TODO Implement neuron-level inhibition (?)
  
  def actionPotential_approximate(self):
    # Action potential - approximate method: Instantaneous rise
    self.potential = action_potential_peak  # depolarization/rising phase (instantaneous)
  
  def actionPotential_accurate(self):
    # Action potential - accurate method: Gradual rise (harder to do in real time)
    #print "[SELF-DEPOLARIZATION]"
    #self.potential += self_depolarization_rate * self.deltaTime  # contant depolarization
    self.potential += (action_potential_peak + 0.02 - self.potential) * 10 * self_depolarization_rate * self.deltaTime  # smoothed depolarization, hackish
  
  actionPotential = actionPotential_approximate  # pick _accurate for more realistic action potential dynamics
  
  def fireActionPotential(self):
    # Transmit action potential to neighbor neurons through axon (TODO introduce transmission delay?)
    #print "Neuron.fireActionPotential() [{}]".format(self.id)  # [debug]
    for synapse in self.synapses:
      synapse.transmitActionPotential(self.timeCurrent)
      
    for synapse in self.gatedSynapses:
      synapse.inhibit(self.timeCurrent)
  
  def plot(self):
    plot((self.timeLastPlotted, self.timeCurrent), (self.potentialLastPlotted, self.potential), self.plotColor)  # [graph]
    self.timeLastPlotted = self.timeCurrent
    self.potentialLastPlotted = self.potential
  
  def __str__(self):
    return "{}: {{ {} }}".format(self.__class__.__name__, ", ".join("{}: {}".format(attr, getattr(self, attr)) for attr in self.__class__._str_attrs))


class NeuronGroup:
  default_bounds = np.float32([[-50.0, -50.0, -5.0], [50.0, 50.0, 5.0]])
  default_distribution = MultivariateNormal(mu=np.float32([0.0, 0.0, 0.0]), cov=(np.float32([400, 400, 4]) * np.identity(3, dtype=np.float32)))
  
  def __init__(self, numNeurons=1000, timeNow=0.0, neuronTypes=[Neuron], bounds=default_bounds, distribution=default_distribution):
    self.numNeurons = numNeurons
    self.timeNow = timeNow
    self.neuronTypes = neuronTypes
    self.bounds = bounds
    self.distribution = distribution
    self.isConnected = False
    
    self.logger = logging.getLogger(__name__)
    self.logger.info("Creating {}".format(self))
    self.logger.debug("Bounds: x: {}, y: {}, z: {}".format(self.bounds[:,0], self.bounds[:,1], self.bounds[:,2]))
    self.neuronLocations = []
    if isinstance(self.distribution, MultivariateNormal):
      #self.logger.debug("Distribution: mu: {}, cov: {}".format(self.distribution.mu, self.distribution.cov))  # ugly
      self.neuronLocations = np.random.multivariate_normal(self.distribution.mu, self.distribution.cov, self.numNeurons)
    elif isinstance(self.distribution, SymmetricNormal):
      thetas = np.random.uniform(pi, -pi, self.numNeurons)  # symmetric in any direction around Z axis
      rads = np.random.normal(self.distribution.mu, self.distribution.sigma, self.numNeurons)  # varies radially
      xLocs, yLocs = cv2.polarToCart(rads, thetas)
      zLocs = np.repeat(np.float32([self.distribution.center[2]]), self.numNeurons).reshape((self.numNeurons, 1))  # constant z, repeated as a column vector
      #self.logger.debug("SymmetricNormal array shapes:- x: {}, y: {}, z: {}".format(xLocs.shape, yLocs.shape, zLocs.shape))
      self.neuronLocations = np.hstack([self.distribution.center[0] + xLocs, self.distribution.center[1] + yLocs, zLocs])  # build Nx3 numpy array
    elif isinstance(self.distribution, SymmetricLogNormal):
      thetas = np.random.uniform(pi, -pi, self.numNeurons)  # symmetric in any direction around Z axis
      rads = np.random.lognormal(self.distribution.mu, self.distribution.sigma, self.numNeurons)  # varies radially
      xLocs, yLocs = cv2.polarToCart(rads, thetas)
      zLocs = np.repeat(np.float32([self.distribution.center[2]]), self.numNeurons).reshape((self.numNeurons, 1))  # constant z, repeated as a column vector
      #self.logger.debug("SymmetricNormal array shapes:- x: {}, y: {}, z: {}".format(xLocs.shape, yLocs.shape, zLocs.shape))
      self.neuronLocations = np.hstack([self.distribution.center[0] + xLocs, self.distribution.center[1] + yLocs, zLocs])  # build Nx3 numpy array
    else:
      raise ValueError("Unknown distribution type: {}".format(type(self.distribution)))
    # TODO Include (non-central) F distribution (suitable for rods)
    #print "Neuron locations:\n", self.neuronLocations  # [debug]
  
    self.neurons = self.numNeurons * [None]
    # TODO Build spatial index using oct-tree
    self.neuronPlotColors = self.numNeurons * [None]
    for i in xrange(self.numNeurons):
      self.neurons[i] = Neuron(self.neuronLocations[i], self.timeNow)
      self.neuronPlotColors[i] = self.neurons[i].plotColor
  
  def connectWith(self, group, maxConnectionsPerNeuron, growthCone):
    self.growthCone = growthCone  # cache for possible display
    self.numSynapses = 0
    self.numDisconnectedNeurons = 0
    
    # * For each neuron in this group
    for a in self.neurons:
      # ** Find candidate neurons from the other group
      candidates = []
      for b in group.neurons:  # TODO: Sample/optimize
        growthConeScore = growthCone.score(a.location, b.location)
        if growthConeScore > 0.1:
          candidates.append((growthConeScore, b))
      
      # ** Sort candidates based on scores, and pick top n (TODO: Add some probabilistic noise?)
      candidates.sort(key=lambda pair: pair[0], reverse=True)
      for i in xrange(min(maxConnectionsPerNeuron, len(candidates))):
        a.synapseWith(candidates[i][1])  # TODO: Use score as synaptic strength?
      
      self.numSynapses += len(a.synapses)
      if not a.synapses:
        self.numDisconnectedNeurons += 1
    
    self.logger.debug("Pre: {}, post: {}, #synapses: {}, (avg.: {} per pre-neuron), #disconnected: {}".format(len(self.neurons), len(group.neurons), self.numSynapses, float(self.numSynapses) / len(self.neurons), self.numDisconnectedNeurons))
    self.isConnected = True
  
  def plotNeuronLocations3D(self, ax=None, showConnections=True, groupColor=None, connectionColor=None, equalScaleZ=False):
    standalone = False
    if ax is None:
      standalone = True
      fig = figure()
      ax = fig.gca(projection='3d')
    
    ax.scatter(self.neuronLocations[:,0], self.neuronLocations[:,1], self.neuronLocations[:,2], c=(self.neuronPlotColors if groupColor is None else groupColor))
    if showConnections and self.isConnected:
      for n in self.neurons:
        frm = n.location
        to = n.location + self.growthCone.maxLength * self.growthCone.direction
        #ax.plot((frm[0], to[0]), (frm[1], to[1]), (frm[2], to[2]))  # [debug: draw growth cone vector]
        for s in n.synapses:
          ax.plot((n.location[0], s.post.location[0]), (n.location[1], s.post.location[1]), (n.location[2], s.post.location[2]), c=(n.plotColor if connectionColor is None else connectionColor))
    
    if standalone:  # TODO prevent code duplication
      plot_bounds = self.bounds
      plot_sizes = (plot_bounds[1] - plot_bounds[0])
      max_plot_size = max(plot_sizes)
      plot_centers = (plot_bounds[0] + plot_bounds[1]) / 2
      x_bounds = [plot_centers[0] - max_plot_size / 2, plot_centers[0] + max_plot_size / 2]
      y_bounds = [plot_centers[1] - max_plot_size / 2, plot_centers[1] + max_plot_size / 2]
      if equalScaleZ:
        z_bounds = [plot_centers[2] - max_plot_size / 2, plot_centers[2] + max_plot_size / 2]  # Z axis scaled the same way as rest
      else:
        z_bounds =  plot_bounds[:, 2]  # separate scale for Z axis
      ax.auto_scale_xyz(x_bounds, y_bounds, z_bounds)
      ax.set_xlabel("X")
      ax.set_ylabel("Y")
      ax.set_zlabel("Z")
      show()
  
  def __str__(self):
    return "NeuronGroup: {{ numNeurons: {}, neuronTypes: [{}] }}".format(self.numNeurons, ", ".join(t.__name__ for t in self.neuronTypes))
  
  def __repr__(self):
    return "NeuronGroup: {{ numNeurons: {}, neuronTypes: [{}], bounds: {}, distribution: {} }}".format(self.numNeurons, ", ".join(t.__name__ for t in self.neuronTypes), repr(self.bounds), self.distribution)


def plotNeuronGroups(neuronGroups, groupColors=None, showConnections=True, connectionColors=None, equalScaleZ=False):
  if groupColors == None:
    groupColors = [None] * len(neuronGroups)
  if connectionColors == None:
    connectionColors = [None] * len(neuronGroups)
  
  fig = figure()
  ax = fig.gca(projection='3d')  # effectively same as fig.add_subplot(111, projection='3d')
  
  plot_bounds = np.float32([np.repeat(np.inf, 3), np.repeat(-np.inf, 3)])
  for group, groupColor, connectionColor in zip(neuronGroups, groupColors, connectionColors):
    group.plotNeuronLocations3D(ax, showConnections=showConnections, groupColor=groupColor, connectionColor=connectionColor)
    plot_bounds[0, :] = np.minimum(plot_bounds[0], group.bounds[0])
    plot_bounds[1, :] = np.maximum(plot_bounds[1], group.bounds[1])
  
  plot_sizes = (plot_bounds[1] - plot_bounds[0])
  max_plot_size = max(plot_sizes)
  plot_centers = (plot_bounds[0] + plot_bounds[1]) / 2
  x_bounds = [plot_centers[0] - max_plot_size / 2, plot_centers[0] + max_plot_size / 2]
  y_bounds = [plot_centers[1] - max_plot_size / 2, plot_centers[1] + max_plot_size / 2]
  if equalScaleZ:
    z_bounds = [plot_centers[2] - max_plot_size / 2, plot_centers[2] + max_plot_size / 2]  # Z axis scaled the same way as rest
  else:
    z_bounds =  plot_bounds[:, 2]  # separate scale for Z axis
  ax.auto_scale_xyz(x_bounds, y_bounds, z_bounds)
  
  ax.set_aspect('equal')
  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_zlabel("Z")
  subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
  show()


def test_neuronGroup():
  logging.basicConfig(format="%(levelname)s | %(name)s | %(funcName)s() | %(message)s", level=logging.DEBUG)  # sets up basic logging, if it's not already configured
  startTime = time()
  timeNow = 0.0
  group1 = NeuronGroup(numNeurons=1000, timeNow=timeNow)
  group2 = NeuronGroup(numNeurons=500, timeNow=timeNow, bounds=np.float32([[-25.0, -25.0, 7.5], [25.0, 25.0, 12.5]]), distribution=MultivariateNormal(mu=np.float32([0.0, 0.0, 10.0]), cov=(np.float32([400, 400, 4]) * np.identity(3))))
  growthConeDirection = group2.distribution.mu - group1.distribution.mu
  growthConeDirection /= np.linalg.norm(growthConeDirection, ord=2)  # need a unit vector
  group1.connectWith(group2, maxConnectionsPerNeuron=25, growthCone=GrowthCone(growthConeDirection))
  #group2.plotNeuronLocations3D(equalScaleZ=True)  # e.g.: plot a single neuron group
  plotNeuronGroups([group1, group2], groupColors=['b', 'r'], showConnections=True, connectionColors=[None, None], equalScaleZ=True)
  # NOTE: For connectionColors, pass None to draw connection lines with pre-neuron's color; or specify colors explicitly, e.g.: connectionColors=[(0.9, 0.8, 1.0, 0.5), None]


if __name__ == "__main__":
  test_neuronGroup()
