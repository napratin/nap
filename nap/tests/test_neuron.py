from time import time, sleep
from unittest import TestCase
import numpy as np

from nap.neuron import Neuron, threshold_potential, action_potential_peak, action_potential_trough, synaptic_strength, setup_neuron_plot

from matplotlib.pyplot import figure, plot, subplot, subplots_adjust, draw, pause, hold, show, xlim, ylim, title, xlabel, ylabel, axhline
import matplotlib as mpl
mpl.rc('axes', titlesize=22, labelsize=20)
mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)

holdPlots = False  # wait for user to close plot windows?


class TestNeuron(TestCase):
  """Test neuron functions individually and in small groups."""
  
  def setUp(self):
    pass
  
  def tearDown(self):
    pass
  
  def test_single(self, duration=10.0, delay=0.01):
    startTime = time()
    timeNow = 0.0
    n = Neuron((0.0, 0.0, 0.0), timeNow)
    
    # Set up plotting
    figure(figsize = (12, 9))
    hold(True)  # [graph]
    xlim(0.0, duration)  # [graph]
    ylim(action_potential_trough.mu - 0.01, action_potential_peak + 0.02)  # [graph]
    setup_neuron_plot("Stimulated neuron", None, "Membrane potential (V)")
    
    while timeNow <= duration:
      timeNow = time() - startTime
      n.accumulate(np.random.normal(0.001, 0.00025))
      n.update(timeNow)
      n.plot()  # [graph]
      pause(delay)  # [graph]
      #sleep(delay)
    
    if holdPlots:
      # Show plot (and wait till it's closed)
      show()  # [graph]
  
  def test_pair(self, pre_duration=2.0, stimulus_duration=10.0, post_duration=2.0, delay=0.01):
    total_duration = pre_duration + stimulus_duration + post_duration
    stimulus_begin = pre_duration
    stimulus_end = pre_duration + stimulus_duration
    
    startTime = time()
    timeNow = 0.0
    
    n1 = Neuron((0.0, 0.0, 0.0), timeNow)
    n2 = Neuron((0.0, 0.0, 1.0), timeNow)
    n1.synapseWith(n2)
    
    # Set up plotting
    figure(figsize = (12, 9))
    hold(True)  # [graph]
    subplot(211)  # [graph]
    xlim(0.0, total_duration)  # [graph]
    ylim(action_potential_trough.mu - 0.01, action_potential_peak + 0.02)  # [graph]
    setup_neuron_plot("Presynaptic neuron", None, "Membrane potential (V)")
    subplot(212)  # [graph]
    xlim(0.0, total_duration)  # [graph]
    ylim(action_potential_trough.mu - 0.01, action_potential_peak + 0.02)  # [graph]
    setup_neuron_plot("Postsynaptic neuron", "Time (s)", "Membrane potential (V)")
    subplots_adjust(hspace = 0.33)
    
    while timeNow <= total_duration:
      timeNow = time() - startTime
      if stimulus_begin <= timeNow <= stimulus_end:
        n1.accumulate(np.random.normal(0.0035, 0.0005))  # TODO accumulate value based on deltaTime (since last accumulate)
      n1.update(timeNow)
      #print n1.id, n1.timeCurrent, n1.potential  # [log: potential]
      n2.update(timeNow)
      #print n2.id, n2.timeCurrent, n2.potential  # [log: potential]
      
      subplot(211)  # [graph]
      n1.plot()  # [graph]
      subplot(212)  # [graph]
      n2.plot()  # [graph]
      pause(delay)  # [graph]
      #sleep(delay)
    
    if holdPlots:
      show()  # [graph]


  def test_gatekeeper(self, total_duration=14.0, stimulus_period=(2.0, 12.0), gate_period=(5.0, 8.0), delay=0.01):
    startTime = time()
    timeNow = 0.0
    
    n1 = Neuron((0.0, 0.0, 0.0), timeNow)
    n2 = Neuron((0.0, 0.0, 1.0), timeNow)
    g1 = Neuron((-1.0, -1.0, 1.0), timeNow)  # gatekeeper
    n1.synapseWith(n2, None, g1)  # auto-initialize synaptic strength
    
    # Set up plotting
    figure(figsize = (12, 9))
    hold(True)  # [graph]
    ax = subplot(311)  # [graph]
    xlim(0.0, total_duration)  # [graph]
    ylim(action_potential_trough.mu - 0.01, action_potential_peak + 0.01)  # [graph]
    setup_neuron_plot("Presynaptic neuron", None, None)
    ax.get_xaxis().set_ticklabels([])
    ax = subplot(312)  # [graph]
    xlim(0.0, total_duration)  # [graph]
    ylim(action_potential_trough.mu - 0.01, action_potential_peak + 0.01)  # [graph]
    setup_neuron_plot("Postsynaptic neuron", None, "Membrane potential (V)")
    ax.get_xaxis().set_ticklabels([])
    subplot(313)  # [graph]
    xlim(0.0, total_duration)  # [graph]
    ylim(action_potential_trough.mu - 0.01, action_potential_peak + 0.01)  # [graph]
    setup_neuron_plot("Gatekeeper neuron", "Time (s)", None)
    subplots_adjust(hspace = 0.33)
    
    while timeNow <= total_duration:
      timeNow = time() - startTime
      
      if stimulus_period[0] <= timeNow <= stimulus_period[1]:
        n1.accumulate(np.random.normal(0.004, 0.0005))  # TODO accumulate value based on deltaTime (since last accumulate)
      
      if gate_period[0] <= timeNow <= gate_period[1]:
        g1.accumulate(np.random.normal(0.0035, 0.0005))
      
      n1.update(timeNow)
      #print n1.id, n1.timeCurrent, n1.potential  # [log: potential]
      n2.update(timeNow)
      #print n2.id, n2.timeCurrent, n2.potential  # [log: potential]
      g1.update(timeNow)
      
      subplot(311)  # [graph]
      n1.plot()  # [graph]
      subplot(312)  # [graph]
      n2.plot()  # [graph]
      subplot(313)  # [graph]
      g1.plot()  # [graph]
      pause(delay)  # [graph]
      #sleep(delay)
    
    if holdPlots:
      show()  # [graph]


  def test_inhibitor(self, total_duration=14.0, stimulus_period=(2.0, 12.0), inhibition_period=(5.0, 8.0), delay=0.01):
    startTime = time()
    timeNow = 0.0
    
    n1 = Neuron((0.0, 0.0, 0.0), timeNow)
    n2 = Neuron((0.0, 0.0, 1.0), timeNow)
    i1 = Neuron((-1.0, -1.0, 1.0), timeNow)  # inhibitor
    n1.synapseWith(n2)  # no synaptic gating
    i1.synapseWith(n2, -np.random.normal(synaptic_strength.mu, synaptic_strength.sigma))  # inhibitory synapse
    
    # Set up plotting
    figure(figsize = (12, 9))
    hold(True)  # [graph]
    subplot(311)  # [graph]
    xlim(0.0, total_duration)  # [graph]
    ylim(action_potential_trough.mu - 0.01, action_potential_peak + 0.01)  # [graph]
    setup_neuron_plot("Neuron " + str(n1.id))
    subplot(312)  # [graph]
    xlim(0.0, total_duration)  # [graph]
    ylim(action_potential_trough.mu - 0.01, action_potential_peak + 0.01)  # [graph]
    setup_neuron_plot("Neuron " + str(n2.id))
    subplot(313)  # [graph]
    xlim(0.0, total_duration)  # [graph]
    ylim(action_potential_trough.mu - 0.01, action_potential_peak + 0.01)  # [graph]
    setup_neuron_plot("Neuron " + str(i1.id) + " (inhibitor)")
    subplots_adjust(hspace = 0.33)
    
    while timeNow <= total_duration:
      timeNow = time() - startTime
      
      if stimulus_period[0] <= timeNow <= stimulus_period[1]:
        n1.accumulate(np.random.normal(0.004, 0.0005))  # TODO accumulate value based on deltaTime (since last accumulate)
      
      if inhibition_period[0] <= timeNow <= inhibition_period[1]:
        i1.accumulate(np.random.normal(0.0035, 0.0005))
      
      n1.update(timeNow)
      #print n1.id, n1.timeCurrent, n1.potential  # [log: potential]
      n2.update(timeNow)
      #print n2.id, n2.timeCurrent, n2.potential  # [log: potential]
      i1.update(timeNow)
      
      subplot(311)  # [graph]
      n1.plot()  # [graph]
      subplot(312)  # [graph]
      n2.plot()  # [graph]
      subplot(313)  # [graph]
      i1.plot()  # [graph]
      pause(delay)  # [graph]
      #sleep(delay)
    
    if holdPlots:
      show()  # [graph]
