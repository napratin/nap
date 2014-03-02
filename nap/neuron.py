"""A neuron model based on a biologically inspired neuronal architecture."""

import logging
from time import time
from math import pi, exp, log, sqrt
import random
from collections import namedtuple, deque
import numpy as np
import cv2

from matplotlib.pyplot import figure, plot, axis, show, subplots_adjust
from mpl_toolkits.mplot3d import Axes3D

from .util.quadtree import Rect, QuadTree

# Different neuron distributions
Uniform = namedtuple('Uniform', ['low', 'high'])  # a uniform distribution over the half-open interval [low, high)
MultivariateUniform = namedtuple('MultivariateUniform', ['lows', 'highs'])  # a uniform distribution in multiple dimensions
Normal = namedtuple('Normal', ['mu', 'sigma'])  # a normal distribution, defined by mean (mu) and std. dev. (sigma)
MultivariateNormal = namedtuple('MultivariateNormal', ['mu', 'cov'])  # a multivariate normal distribution, defined by mean vector (mu) of length N and covariance matrix (cov) of size NxN
SymmetricNormal = namedtuple('SymmetricNormal', Normal._fields + ('center',))
SymmetricLogNormal = namedtuple('SymmetricLogNormal', SymmetricNormal._fields)

# Neuron membrane potential levels
threshold_potential = -0.055  # volts; level that triggers action potential
action_potential_peak = 0.04  # volts; maximum potential reached during an action potential event
action_potential_trough = Normal(-0.08, 0.001)  # volts; minimum potential reached due to hyperpolarization during an action potential event

# Synaptic strength distribution (TODO find out from literature what these values should be)
synaptic_strength = Normal(0.011, 0.001)  # essentially volts (mean, s.d.); potential transmitted to post-synaptic neuron when the pre-synaptic neuron undergoes an action potential event

# Timing, decay and dynamic action potential parameters
self_depolarization_rate = 0.75  # volts per sec.; rate at which potential rises during an action potential event
refractory_period = 0.1  # secs.; minimum time between two action potentials (should be an emergent effect when simulating action potential in detail)
min_update_time = 0.025  # secs.; minimum time between updates
synapse_inhibition_period = 0.5  # secs.; duration for which the effect of inhibition lasts at a synapse
neuron_inhibition_period = 1.25  #  secs.; duration for which the effect of inhibition lasts in a neuron

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
  
  def getTerminationRect(self, anchor):
    """Given an anchor, return the approximate region of termination as a rectangle."""
    radius = self.maxLength * self.spreadFactor
    return Rect(anchor[0] - radius, anchor[1] - radius, anchor[0] + radius, anchor[1] + radius)
  
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
    # TODO Implement learning/boosting of synaptic strength
    
    # Synapse-level inhibition
    self.isInhibited = False
    self.uninhibitAt = -1.0
  
  def transmitActionPotential(self, timeNow):
    if self.isInhibited and self.uninhibitAt <= timeNow:
      self.isInhibited = False
    #print "Synapse.transmitActionPotential(): timeNow = {}, self.uninhibitAt = {} [{}]".format(timeNow, self.uninhibitAt, "INHIBITED" if self.isInhibited else "UNINHIBITED")
    
    if not self.isInhibited:
      self.post.accumulate(self.strength)
  
  def transmitGradedPotential(self, timeNow):
    if self.isInhibited and self.uninhibitAt <= timeNow:
      self.isInhibited = False
    #print "Synapse.transmitGradedPotential(): timeNow = {}, self.uninhibitAt = {} [{}]".format(timeNow, self.uninhibitAt, "INHIBITED" if self.isInhibited else "UNINHIBITED")
    
    if not self.isInhibited:
      #self.post.accumulate((self.pre.potential - self.pre.resting_potential.mu))
      self.post.accumulate((self.pre.potential - self.pre.resting_potential.mu) * self.strength)
      # TODO Figure out how to quantize graded potential transmission
  
  def inhibit(self, timeNow, duration=synapse_inhibition_period):
    #print "Synapse.inhibit(): timeNow = {}, duration = {}".format(timeNow, duration)
    self.isInhibited = True
    self.uninhibitAt = timeNow + duration


class Neuron:
  """A simple excitable neuron cell with synaptic connections, potential accumulation and decay functionality."""
  
  id_ctr = 0  # auto-incremented counter to assign unique IDs to instances
  _str_attrs = ['id', 'location', 'potential']  # which attributes to include in string representation; subclasses can override this
  
  resting_potential = Normal(-0.07, 0.001)  # volts (mean, s.d.); resting / equillibrium potential
  potential_decay = 1.0  # per-sec.; rate at which potential decays trying to reach equillibrium
  
  p_factor = 1.0  # factor used to scale update probability
  min_p = 0.15  # minimum update probability, to prevent starving; maximum is implicitly 1.0
  
  def __init__(self, location, timeNow):
    self.id = Neuron.id_ctr
    Neuron.id_ctr += 1
    self.location = location  # location in 3-space
    self.timeLastFired = self.timeLastUpdated = self.timeCurrent = timeNow
    self.deltaTime = 0.0
    
    self.potential = np.random.normal(self.resting_potential.mu, self.resting_potential.sigma)  # current membrane potential
    self.potentialLastUpdated = self.potential  # last computed potential, useful for calculating rate of change
    self.potentialAccumulated = 0.0  # potential accumulated from synaptic inputs
    self.p = np.random.uniform(0.0, 0.25)  # update probability: [0, 1] (actually, no need to clip at 1)
    
    self.synapses = list()
    self.gatedNeurons = list()
    self.gatedSynapses = list()
    
    # Neuron-level inhibition
    self.isInhibited = False
    self.uninhibitAt = -1.0
    
    self.timeLastPlotted = self.timeCurrent  # [graph]
    self.potentialLastPlotted = self.potential  # [graph]
    self.plotColor = plot_colors[self.id % len(plot_colors)]  # [graph]
  
  def synapseWith(self, neuron, strength=None, gatekeeper=None):
    s = Synapse(self, neuron, strength, gatekeeper)
    self.synapses.append(s)
    if gatekeeper is not None:
      gatekeeper.gateSynapse(s)
  
  def gateNeuron(self, neuron):
    self.gatedNeurons.append(neuron)
  
  def gateSynapse(self, synapse):
    self.gatedSynapses.append(synapse)
  
  def accumulate(self, deltaPotential):
    self.potentialAccumulated += deltaPotential
  
  def updateWithP(self, timeNow):
    if self.p >= random.random():
      self.update(timeNow)
  
  def update(self, timeNow):
    if self.isInhibited and self.uninhibitAt <= timeNow:
      self.isInhibited = False
      # NOTE potentialAccumulated is not reset here so that it can start responding immediately
    
    if self.isInhibited:
      self.potentialAccumulated = 0.0  # lose any potential gained in this period
    else:
      self.timeCurrent = timeNow
      self.deltaTime = self.timeCurrent - self.timeLastUpdated
      if self.deltaTime < min_update_time:
        return
      
      self.updatePotential()
      self.updateP()
      self.potentialLastUpdated = self.potential
      self.timeLastUpdated = self.timeCurrent
  
  def updatePotential(self):
    # Fire action potential, if we've reached peak
    if self.potential >= action_potential_peak:
      self.fireActionPotential()
      self.timeLastFired = self.timeCurrent
      self.potential = np.random.normal(action_potential_trough.mu, action_potential_trough.sigma)  # repolarization/falling phase (instantaneous)
    
    # Decay potential
    #self.potential = self.resting_potential.mu + (self.potential - self.resting_potential.mu) * exp(-self.potential_decay * self.deltaTime)  # exponential decay
    self.potential -= self.potential_decay * (self.potential - self.resting_potential.mu) * self.deltaTime  # approximated exponential decay
    
    # Accumulate/integrate incoming potentials
    self.potential += self.potentialAccumulated  # integrate signals accumulated from neighbors
    self.potentialAccumulated = 0.0  # reset accumulator (don't want to double count!)
    
    # Check for action potential event
    if self.potential > threshold_potential and (self.timeCurrent - self.timeLastFired) >= refractory_period:
      self.actionPotential()
    
    #print self.id, self.timeCurrent, self.potential  # [log: potential]
    # TODO This is the ideal point to gather potential observation; "Fire action potential" step should come immediately after this (instead of at the beginning of updatePotential) in order to prevent any posible delays
    # TODO Implement neuron-level inhibition (?)
  
  def updateP(self):
    self.p = np.clip(self.p_factor * abs(self.potential - self.potentialLastUpdated) / self.deltaTime, self.min_p, 1.0)
    #if self.p > 1.0: self.p = 1.0  # no need to clip at 1 because of the way this is used
  
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
    # Fire action potential to neighbor neurons through axon (TODO introduce transmission delay?)
    #print "Neuron.fireActionPotential() [{}]".format(self.id)  # [debug]
    for synapse in self.synapses:
      synapse.transmitActionPotential(self.timeCurrent)
    
    for neuron in self.gatedNeurons:
      neuron.inhibit(self.timeCurrent)
    
    for synapse in self.gatedSynapses:
      synapse.inhibit(self.timeCurrent)
  
  def sendGradedPotential(self):
    # Send graded potential to neighbor neurons through axon
    #print "Neuron.sendGradedPotential() [{}]".format(self.id)  # [debug]
    for synapse in self.synapses:
      synapse.transmitGradedPotential(self.timeCurrent)
  
  def inhibit(self, timeNow, duration=neuron_inhibition_period):
    #print "Neuron.inhibit(): timeNow = {}, duration = {}".format(timeNow, duration)
    self.isInhibited = True
    self.uninhibitAt = timeNow + duration
    self.potentialLastUpdated = self.potential = np.random.normal(self.resting_potential.mu, self.resting_potential.sigma)  # reset potential
    self.timeLastUpdated = timeNow
  
  def plot(self):
    plot((self.timeLastPlotted, self.timeCurrent), (self.potentialLastPlotted, self.potential), self.plotColor)  # [graph]
    self.timeLastPlotted = self.timeCurrent
    self.potentialLastPlotted = self.potential
  
  def __str__(self):
    return "{}: {{ {} }}".format(self.__class__.__name__, ", ".join("{}: {}".format(attr, getattr(self, attr)) for attr in self.__class__._str_attrs))


class Population:
  default_bounds = np.float32([[-50.0, -50.0, -5.0], [50.0, 50.0, 5.0]])
  default_distribution = MultivariateNormal(mu=np.float32([0.0, 0.0, 0.0]), cov=(np.float32([400, 400, 4]) * np.identity(3, dtype=np.float32)))
  
  def __init__(self, numNeurons=1000, timeNow=0.0, neuronTypes=[Neuron], bounds=default_bounds, distribution=default_distribution, **kwargs):
    self.numNeurons = numNeurons
    self.timeNow = timeNow
    self.neuronTypes = neuronTypes
    self.bounds = bounds
    self.center = (self.bounds[0] + self.bounds[1]) / 2
    self.distribution = distribution
    self.isConnected = False
    
    self.logger = logging.getLogger(__name__)
    self.logger.info("Creating {}".format(self))
    self.logger.debug("Bounds: x: {}, y: {}, z: {}".format(self.bounds[:,0], self.bounds[:,1], self.bounds[:,2]))
    
    # * Designate neuron locations
    self.neuronLocations = []
    if isinstance(self.distribution, MultivariateUniform):
      # NOTE self.distribution has to be a 3-dimensional MultivariateUniform, even if the third dimension is a constant (low=high)
      self.neuronLocations = np.column_stack([
        np.random.uniform(self.distribution.lows[0], self.distribution.highs[0], self.numNeurons),
        np.random.uniform(self.distribution.lows[1], self.distribution.highs[1], self.numNeurons),
        np.random.uniform(self.distribution.lows[2], self.distribution.highs[2], self.numNeurons)])
      #self.logger.debug("MultivariateUniform array shape: {}".format(self.neuronLocations.shape))
    elif isinstance(self.distribution, MultivariateNormal):
      #self.logger.debug("Distribution: mu: {}, cov: {}".format(self.distribution.mu, self.distribution.cov))  # ugly
      self.neuronLocations = np.random.multivariate_normal(self.distribution.mu, self.distribution.cov, self.numNeurons)
    elif isinstance(self.distribution, SymmetricNormal):
      thetas = np.random.uniform(pi, -pi, self.numNeurons)  # symmetric in any direction around Z axis
      rads = np.random.normal(self.distribution.mu, self.distribution.sigma, self.numNeurons)  # varies radially
      xLocs, yLocs = cv2.polarToCart(rads, thetas)
      zLocs = np.repeat(np.float32([self.distribution.center[2]]), self.numNeurons).reshape((self.numNeurons, 1))  # constant z, repeated as a column vector
      #self.logger.debug("SymmetricNormal array shapes:- x: {}, y: {}, z: {}".format(xLocs.shape, yLocs.shape, zLocs.shape))
      self.neuronLocations = np.column_stack([
        self.distribution.center[0] + xLocs,
        self.distribution.center[1] + yLocs,
        zLocs])  # build Nx3 numpy array
    elif isinstance(self.distribution, SymmetricLogNormal):
      thetas = np.random.uniform(pi, -pi, self.numNeurons)  # symmetric in any direction around Z axis
      rads = np.random.lognormal(self.distribution.mu, self.distribution.sigma, self.numNeurons)  # varies radially
      xLocs, yLocs = cv2.polarToCart(rads, thetas)
      zLocs = np.repeat(np.float32([self.distribution.center[2]]), self.numNeurons).reshape((self.numNeurons, 1))  # constant z, repeated as a column vector
      #self.logger.debug("SymmetricLogNormal array shapes:- x: {}, y: {}, z: {}".format(xLocs.shape, yLocs.shape, zLocs.shape))
      self.neuronLocations = np.column_stack([
        self.distribution.center[0] + xLocs,
        self.distribution.center[1] + yLocs,
        zLocs])  # build Nx3 numpy array
    else:
      raise ValueError("Unknown distribution type: {}".format(type(self.distribution)))
    # TODO Include (non-central) F distribution (suitable for rods)
    
    # Clip (clamp) neuron locations that are outside bounds
    np.clip(self.neuronLocations[:, 0], self.bounds[0, 0], self.bounds[1, 0], out=self.neuronLocations[:, 0])
    np.clip(self.neuronLocations[:, 1], self.bounds[0, 1], self.bounds[1, 1], out=self.neuronLocations[:, 1])
    #print "Out-of-bounds neuron locations:", [loc for loc in self.neuronLocations if not ((self.bounds[0, 0] <= loc[0] <= self.bounds[1, 0]) and (self.bounds[0, 1] <= loc[1] <= self.bounds[1, 1]))]  # [debug]
    
    #print "Neuron locations:\n", self.neuronLocations  # [debug]
    
    # * Create neurons
    self.neurons = self.numNeurons * [None]
    self.neuronPlotColors = self.numNeurons * [None]
    for i in xrange(self.numNeurons):
      self.neurons[i] = random.choice(self.neuronTypes)(self.neuronLocations[i], self.timeNow, **kwargs)
      self.neuronPlotColors[i] = self.neurons[i].plotColor
    
    # * Build spatial index using quadtree (assuming neurons are roughly in a layer)
    boundingRect = (self.bounds[0, 0], self.bounds[0, 1], self.bounds[1, 0], self.bounds[1, 1])
    self.qtree = QuadTree(self.neurons, depth=int(log(self.numNeurons, 2)), bounding_rect=boundingRect)
  
  # TODO Move this to Projection
  def connectWith(self, population, maxConnectionsPerNeuron, growthCone=None, allowSelfConnections=False):
    if growthCone is not None:
      self.growthCone = growthCone
    else:
      growthConeDirection = population.center - self.center
      growthConeLength = np.linalg.norm(growthConeDirection, ord=2)
      growthConeDirection /= growthConeLength  # need a unit vector
      self.growthCone = GrowthCone(growthConeDirection, maxLength=growthConeLength * 2.0, spreadFactor=1)
    
    self.numSynapses = 0
    self.numDisconnectedNeurons = 0
    
    # * For each neuron in this population
    for a in self.neurons:
      # ** Compute search rectangle in target population to select candidate neurons
      rect = self.growthCone.getTerminationRect(a.location)
      
      # ** Find candidate neurons from the other population
      candidates = []
      for b in population.qtree.hit(rect):  # optimized spatial range query
        if a == b and not allowSelfConnections: continue  # skip connecting to self, in case target population is same as this population
        growthConeScore = self.growthCone.score(a.location, b.location)
        if growthConeScore > 0.1:
          candidates.append((growthConeScore, b))
      
      # ** Sort candidates based on scores, and pick top n (TODO: Add some probabilistic noise?)
      candidates.sort(key=lambda pair: pair[0], reverse=True)
      for i in xrange(min(maxConnectionsPerNeuron, len(candidates))):
        a.synapseWith(candidates[i][1])  # TODO: Use score as synaptic strength?
      
      self.numSynapses += len(a.synapses)
      if not a.synapses:
        self.numDisconnectedNeurons += 1
    
    self.logger.debug("Pre: {}, post: {}, #synapses: {}, (avg.: {} per pre-neuron), #disconnected: {}".format(len(self.neurons), len(population.neurons), self.numSynapses, float(self.numSynapses) / len(self.neurons), self.numDisconnectedNeurons))
    self.isConnected = True
  
  def plotNeuronLocations3D(self, ax=None, showConnections=True, populationColor=None, connectionColor=None, equalScaleZ=False):
    standalone = False
    if ax is None:
      standalone = True
      fig = figure()
      ax = fig.gca(projection='3d')
    
    ax.scatter(self.neuronLocations[:,0], self.neuronLocations[:,1], self.neuronLocations[:,2], c=(self.neuronPlotColors if populationColor is None else populationColor))
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
    return "Population: {{ numNeurons: {}, neuronTypes: [{}] }}".format(self.numNeurons, ", ".join(t.__name__ for t in self.neuronTypes))
  
  def __repr__(self):
    return "Population: {{ numNeurons: {}, neuronTypes: [{}], bounds: {}, distribution: {} }}".format(self.numNeurons, ", ".join(t.__name__ for t in self.neuronTypes), repr(self.bounds), self.distribution)


def plotPopulations(populations, populationColors=None, showConnections=True, connectionColors=None, equalScaleZ=False):
  if populationColors == None:
    populationColors = [None] * len(populations)
  if connectionColors == None:
    connectionColors = [None] * len(populations)
  
  fig = figure()
  ax = fig.gca(projection='3d')  # effectively same as fig.add_subplot(111, projection='3d')
  
  plot_bounds = np.float32([np.repeat(np.inf, 3), np.repeat(-np.inf, 3)])
  for population, populationColor, connectionColor in zip(populations, populationColors, connectionColors):
    population.plotNeuronLocations3D(ax, showConnections=showConnections, populationColor=populationColor, connectionColor=connectionColor)
    plot_bounds[0, :] = np.minimum(plot_bounds[0], population.bounds[0])
    plot_bounds[1, :] = np.maximum(plot_bounds[1], population.bounds[1])
  
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


def test_population():
  logging.basicConfig(format="%(levelname)s | %(name)s | %(funcName)s() | %(message)s", level=logging.DEBUG)  # sets up basic logging, if it's not already configured
  startTime = time()
  timeNow = 0.0
  population1 = Population(numNeurons=1000, timeNow=timeNow)
  population2 = Population(numNeurons=500, timeNow=timeNow, bounds=np.float32([[-25.0, -25.0, 7.5], [25.0, 25.0, 12.5]]), distribution=MultivariateNormal(mu=np.float32([0.0, 0.0, 10.0]), cov=(np.float32([400, 400, 4]) * np.identity(3))))
  growthConeDirection = population2.distribution.mu - population1.distribution.mu
  growthConeDirection /= np.linalg.norm(growthConeDirection, ord=2)  # need a unit vector
  population1.connectWith(population2, maxConnectionsPerNeuron=25, growthCone=GrowthCone(growthConeDirection))
  #population2.plotNeuronLocations3D(equalScaleZ=True)  # e.g.: plot a single neuron population
  plotPopulations([population1, population2], populationColors=['b', 'r'], showConnections=True, connectionColors=[None, None], equalScaleZ=True)
  # NOTE: For connectionColors, pass None to draw connection lines with pre-neuron's color; or specify colors explicitly, e.g.: connectionColors=[(0.9, 0.8, 1.0, 0.5), None]


if __name__ == "__main__":
  test_population()
