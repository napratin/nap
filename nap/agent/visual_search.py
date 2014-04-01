"""Visual search using features from visual system and ocular motion."""

import logging
import argparse

from lumos.context import Context
from lumos.input import InputRunner
from lumos import rpc

from ..vision.visual_system import VisualSystem, FeatureManager, default_feature_weight, default_feature_weight_rest


class VisualSearchAgent(object):
  """A simple visual search agent that scans input stream for locations with desired features."""
  
  image_size = VisualSystem.default_image_size  # size of retina to project on
  
  def __init__(self):
    # * Create application context, passing in custom arguments, and get a logger
    argParser = argparse.ArgumentParser(add_help=False)
    argParser.add_argument('--features', type=str, default=None, help="features to look for, comma separated")
    self.context = Context.createInstance(description="Visual search agent", parent_argparsers=[argParser])
    self.logger = logging.getLogger(self.__class__.__name__)
    
    # * Parse arguments
    self.features = self.context.options.features.split(',') if self.context.options.features is not None else []
    self.featureWeights = dict()
    for feature in self.features:
      if ':' in feature:  # check for explicit weights, e.g. RG:0.8,BY:0.75
        try:
          featureSpec = feature.split(':')
          self.featureWeights[featureSpec[0].strip()] = float(featureSpec[1].strip())
        except Exception as e:
          self.logger.warn("Invalid feature specification '%s': %s", feature, e)
      else:  # use default weight
        self.featureWeights[feature.strip()] = default_feature_weight
    if 'rest' not in self.featureWeights:
      self.featureWeights['rest'] = default_feature_weight_rest  # explicitly specify rest, otherwise previous weights will remain
    self.logger.info("Searching with feature weights: %s", self.featureWeights)
    
    # * Create systems and associated managers
    self.context.update()  # get fresh time
    self.visSys = VisualSystem(imageSize=self.image_size, timeNow=self.context.timeNow, showMonitor=False)
    self.visMan = FeatureManager(self.visSys)
    # TODO: Design a better way to share systems/managers (every system has a parent/containing agent?)
    
    # * Export RPC calls, if enabled
    if self.context.isRPCEnabled:
      self.logger.info("Exporting RPC calls")
      rpc.export(self.visSys)
      rpc.export(self.visMan)
      rpc.refresh()  # Context is expected to have started RPC server
  
  def run(self):
    # * Set visual system buffers and send intent
    self.visSys.setBuffer('weights', self.featureWeights)
    self.visSys.setBuffer('intent', 'find')
    
    # * Run vision manager and ocular motion system
    runner = InputRunner(self.visMan)
    
    while runner.update():  # should update context time
      pass
    
    runner.cleanUp()


if __name__ == "__main__":
  VisualSearchAgent().run()
