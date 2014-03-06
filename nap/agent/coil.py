"""Agent to process the COIL-100 dataset using nap."""

import os
import logging
import argparse

from lumos.context import Context
from lumos.util import Enum
from lumos.input import run

from ..vision.visual_system import VisualSystem, VisionManager


class COILManager(VisionManager):
  """A visual system manager for processing a single COIL-100 image."""
  
  State = Enum(('NONE', 'UNSTABLE', 'STABLE', 'DONE'))
  state_duration_unstable = 5.0  # seconds to spend in unstable state before transitioning (when things become stable)
  state_duration_stable = 5.0  # seconds to spend in stable state before transitioning (avoid short stability periods)
  
  def __init__(self, visualSystem=None):
    self.visualSystem = visualSystem if visualSystem is not None else VisualSystem()  # TODO remove this once VisionManager (or its ancestor caches visualSystem in __init__)
    VisionManager.__init__(self, self.visualSystem)
    self.state = self.State.NONE
    self.timeStateChange = -1.0
  
  def initialize(self, imageIn, timeNow):
    VisionManager.initialize(self, imageIn, timeNow)
    self.logger.debug("COILManager initialized")
    self.transition(self.State.UNSTABLE, timeNow)
  
  def process(self, imageIn, timeNow):
    keepRunning, imageOut = VisionManager.process(self, imageIn, timeNow)
    
    # TODO Compute featureVector mean and variance over a moving window
    #self.visualSystem.featureVector
    
    # TODO Change state according to feature vector values
    if self.state == self.State.UNSTABLE and (timeNow - self.timeStateChange) > self.state_duration_unstable:
      self.transition(self.State.STABLE, timeNow)
    elif self.state == self.State.STABLE and (timeNow - self.timeStateChange) > self.state_duration_stable:
      self.transition(self.State.DONE, timeNow)
      return False, imageOut  # Return False when done
    
    return keepRunning, imageOut
  
  def transition(self, next_state, timeNow):
    self.logger.debug("[{:.2f}] Transitioning from {} to {} state".format(timeNow, self.State.toString(self.state), self.State.toString(next_state)))
    self.state = next_state
    self.timeStateChange = timeNow


class COILAgent(object):
  image_size = (200, 200)  # size of the retina on which images are projected, can be larger or smaller than actual size of images
  input_file_prefix = "obj"
  input_file_sep = "__"
  input_file_ext = "png"
  
  def __init__(self):
    # * Create application context, passing in custom arguments, and get a logger
    argParser = argparse.ArgumentParser(add_help=False)
    #argParser.add_argument('--in', type=str, default="coil-100", help="path to directory containing input images")  # use input_source as directory; default to current directory
    argParser.add_argument('--out', type=str, default="out", help="path to output directory")  # should this be a common parameter in Context?
    argParser.add_argument('--obj', type=str, default="1,101,1", required=True, help="object ID range, right-open interval <start>,<stop>,<step> (no spaces)")
    argParser.add_argument('--view', type=str, default="0,360,5", required=True, help="view angle range in degrees, right-open interval <start>,<stop>,<step> (no spaces)")
    self.context = Context.createInstance(description="COIL-100 image dataset processor", parent_argparsers=[argParser])  # TODO how to gather arg parsers from other interested parties?
    self.logger = logging.getLogger(__name__)
    
    # * Parse arguments
    self.inDir = self.context.options.input_source  # will be an absolute path
    #assert os.path.isdir(self.inDir), "Invalid input directory \"{}\"".format(self.inDir)
    self.outDir = self.context.options.out  # just for convenience
    #assert os.path.isdir(self.outDir), "Invalid output directory \"{}\"".format(self.outDir)  # TODO create output directory if it doesn't exist
    self.objRange = xrange(*(int(x) for x in self.context.options.obj.split(',')))
    self.viewRange = xrange(*(int(x) for x in self.context.options.view.split(',')))
    
    # * Create an instance of VisualSystem (VisionManager instance needs to be created per image; TODO change this when InputDevice is able to deal with sequence of separate images)
    self.context.update()  # get fresh time
    self.visualSystem = VisualSystem(imageSize=self.image_size, timeNow=self.context.timeNow, showMonitor=False)
    self.manager = COILManager(self.visualSystem)
  
  def start(self):
    if self.context.isDir:  # input source is a directory
      # * Run visual input using manager, looping over all specified object images
      for obj in self.objRange:
        for view in self.viewRange:
          # ** Build image file path from object ID and view angle
          input_file = os.path.join(self.inDir, "{}{}{}{}.{}".format(self.input_file_prefix, obj, self.input_file_sep, view, self.input_file_ext))
          #assert os.path.exists(input_file), "Input file \"{}\" doesn't exist".format(input_file)
          if not os.path.exists(input_file):
            self.logger.warn("Input file \"{}\" doesn't exist".format(input_file))
            continue
          self.logger.info("Input file: {}".format(input_file))
          
          # ** Modify context to set image file as input source, and run it through the visual system
          self.context.options.input_source = input_file
          self.context.isImage = True
          print "Running..."
          run(self.manager, resetContextTime=False)  # use the same manager so that visual system is only created once
    else:
      run(self.manager, resetContextTime=False)  # run on the sole input source (image or video)


if __name__ == "__main__":
  COILAgent().start()
