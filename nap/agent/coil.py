"""Agent to process the COIL-100 dataset using nap."""

import os
import logging
import argparse
from datetime import datetime
import numpy as np

from lumos.context import Context
from lumos.input import run

from ..vision.visual_system import VisualSystem, FeatureManager


class COILManager(FeatureManager):
  """A visual system manager for processing a single COIL-100 image."""
  
  # Override some FeatureManager parameters
  min_duration_unstable = 3.0
  
  def initialize(self, imageIn, timeNow):
    FeatureManager.initialize(self, imageIn, timeNow)
    
    # Configure visual system to use equal feature weights and hold gaze at a fixed location (default center)
    self.visualSystem.setBuffer('weights', { 'rest': 1.0 })
    self.visualSystem.setBuffer('intent', 'hold')
    # TODO: Use a better feature encoding scheme allowing the visual system to scan different parts of the image
  
  def process(self, imageIn, timeNow):
    keepRunning, imageOut = FeatureManager.process(self, imageIn, timeNow)
    
    if self.state == self.State.STABLE:
      self.logger.info("[Final] Mean: {}".format(self.featureVectorMean))
      self.logger.info("[Final] S.D.: {}".format(self.featureVectorSD))
      return False, imageOut  # Return False when done
    
    return keepRunning, imageOut


class COILAgent(object):
  image_size = (256, 256)  # optimal size can be vary depending on foveal distribution, image size and whether eye movements are enabled or not
  input_file_prefix = "obj"
  input_file_sep = "__"
  input_file_ext = "png"
  output_file_prefix = "feat"
  output_file_sep = "_"
  output_file_ext = "dat"
  
  def __init__(self):
    # * Create application context, passing in custom arguments, and get a logger
    argParser = argparse.ArgumentParser(add_help=False)
    #argParser.add_argument('--in', type=str, default="coil-100", help="path to directory containing input images")  # use input_source as directory; default to current directory
    argParser.add_argument('--out', type=str, default=None, help="path to output directory")  # should this be a common parameter in Context?
    argParser.add_argument('--obj', type=str, default="1,101,1", required=False, help="object ID range, right-open interval <start>,<stop>,<step> (no spaces); default: full range")
    argParser.add_argument('--view', type=str, default="0,360,5", required=False, help="view angle range in degrees, right-open interval <start>,<stop>,<step> (no spaces); default: full range")
    self.context = Context.createInstance(description="COIL-100 image dataset processor", parent_argparsers=[argParser])  # TODO how to gather arg parsers from other interested parties?
    self.logger = logging.getLogger(self.__class__.__name__)
    
    # * Parse arguments
    self.inDir = self.context.options.input_source  # should be an absolute path to a dir with COIL images; if it is a file/camera instead, it will be used as sole input
    # TODO also accept wildcards using glob.glob()?
    self.outDir = self.context.options.out  # just for convenience
    self.outFile = None
    if self.outDir is not None:  # TODO otherwise default to some directory?
      if os.path.isdir(self.outDir):
        now = datetime.now()
        outFilepath = os.path.join(self.outDir, "{}{}{}{}{}.{}".format(self.output_file_prefix, self.output_file_sep, now.strftime('%Y-%m-%d'), self.output_file_sep, now.strftime('%H-%M-%S'), self.output_file_ext))
        self.logger.info("Output file: {}".format(outFilepath))
        self.outFile = open(outFilepath, 'w')  # open output file for storing features (TODO use with.. block instead in start()?)
      else:
        self.logger.warn("Invalid output directory \"{}\"; no output will be saved".format(self.outDir))
        self.outDir = None  # TODO create output directory if it doesn't exist
    self.objRange = xrange(*(int(x) for x in self.context.options.obj.split(',')))
    self.viewRange = xrange(*(int(x) for x in self.context.options.view.split(',')))
    
    # * Create visual system and manager
    self.context.update()  # get fresh time
    self.visSys = VisualSystem(imageSize=self.image_size, timeNow=self.context.timeNow, showMonitor=True)
    self.visMan = COILManager(self.visSys)
    
  def run(self):
    if self.outFile is not None:
      self.outFile.write("{}\t{}\t{}\t{}\n".format('obj', 'view', '\t'.join(["{}_mean".format(label) for label in self.visSys.featureLabels]), '\t'.join(["{}_sd".format(label) for label in self.visSys.featureLabels])))
    
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
          run(self.visMan, resetContextTime=False)  # use the same manager so that visual system is only created once
          if self.outFile is not None:
            self.outFile.write("{}\t{}\t{}\t{}\n".format(obj, view, '\t'.join(str(feat_mean) for feat_mean in self.visMan.featureVectorMean), '\t'.join(str(feat_sd) for feat_sd in self.visMan.featureVectorSD)))
    else:
      run(self.visMan, resetContextTime=False)  # run on the sole input source (image or video)
      #if self.outFile is not None:
      #  self.outFile.write("{}\t{}\t{}\t{}\n".format(obj, view, )) # TODO dunno obj & view for single image, what to do?!
    
    if self.outFile is not None:
      self.outFile.close()
      self.logger.info("Output file closed.")


if __name__ == "__main__":
  COILAgent().run()
