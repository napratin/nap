"""Visual search using features from visual system and ocular motion."""

import os
from math import hypot
import logging
import argparse
import numpy as np
import cv2

from lumos.context import Context
from lumos.input import InputRunner
from lumos import rpc
from lumos.net import ImageServer

from ..vision.visual_system import VisualSystem, VisionManager, FeatureManager, default_feature_weight, default_feature_weight_rest


class VisualSearchAgent(object):
  """A simple visual search agent that scans input stream for locations with desired features."""
  
  image_size = (512, 512)  #VisualSystem.default_image_size  # size of retina to project on
  screen_background = np.uint8([0, 0, 0])  #VisionManager.default_screen_background
  
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
    self.visMan = VisionManager(self.visSys, screen_background=self.screen_background)
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
    
    self.context.resetTime()
    while runner.update():  # should update context time
      if not self.update():
        break
    
    runner.cleanUp()
  
  def update(self):
    """Subclasses should override this to implement per-iteration behavior."""
    return True  # return False to end run


class ZelinksyFinder(VisualSearchAgent):
  """A visual search agent that tries to find a target in a field of distractors (as per Zelinsky et al. 1995, 1997)."""
  
  default_target = 'Q'  # 'O' or 'Q' depending on condition
  default_num_stimuli = 5  # 5 or 17 depending on condition 
  
  #pattern_size = (48, 48)  # size of pattern image [unused]
  #o_radius = 16  # radius of O pattern [unused]
  
  # NOTE: Pathnames relative to root of repository
  fixcross_pattern_file = "res/data/visual-search/zelinsky-patterns/fixcross-pattern.png"
  o_pattern_file = "res/data/visual-search/zelinsky-patterns/o-pattern.png"
  q_pattern_file = "res/data/visual-search/zelinsky-patterns/q-pattern.png"
  
  detect_fixation = False  # TODO: complete fixation cross functionality (end trial, move to next), then enable this
  
  # TODO: Use a dict of patterns to evaluate, one of them target, one of them fixation cross?
  
  # TODO: These need to be updated according to fovea and pattern size
  max_match_sqdiff = 0.02  # max value of SQDIFF matching to be counted as a valid match
  min_confidence_sqdiff = 0.008  # min desired difference between activations for target and distractor (to avoid jumping to conclusion when there is confusion)
  
  def __init__(self, target=default_target, numStimuli=default_num_stimuli, featureChannel='V'):
    VisualSearchAgent.__init__(self)
    self.target = target
    self.featureChannel = featureChannel
    self.numStimuli = numStimuli
    
    # * Configure visual system as needed (shorter times for fast completion goal, may result in some inaccuracy)
    self.visSys.max_free_duration = 0.25
    self.visSys.max_fixation_duration = 1.0
    self.visSys.max_hold_duration = 2.0
    self.visSys.min_good_salience = 0.2  # this task is generally very low-salience
    #self.visSys.min_saccade_salience = 0.1
    
    # * Initialize shape patterns
    # ** Generate  # [unused]
    #self.o_pattern = np.zeros(self.pattern_size, dtype=np.float32)  # [unused]
    #self.q_pattern = np.zeros(self.pattern_size, dtype=np.float32)  # [unused]
    #cv2.circle(self.o_pattern, (self.pattern_size[1] / 2, self.pattern_size[0] / 2), self.o_radius, 1.0, 1)  # [unused]
    # ** Load from files (NOTE: num channels must match selected feature channel)
    self.fixcross_pattern = cv2.imread(self.fixcross_pattern_file, 0)  # pass flags=0 for grayscale
    self.o_pattern = cv2.imread(self.o_pattern_file, 0)  # pass flags=0 for grayscale
    self.q_pattern = cv2.imread(self.q_pattern_file, 0)  # pass flags=0 for grayscale
    #cv2.imshow("Fixation cross pattern", self.fixcross_pattern)
    #cv2.imshow("O pattern", self.o_pattern)
    #cv2.imshow("Q pattern", self.q_pattern)
    
    # * Initialize matching-related objects
    self.numDistractorsSeen = 0  # if numDistractorsSeen >= numStimuli, no target is present
    self.newFixation = False
    self.processFixation = False  # used to prevent repeatedly processing a fixation period once a conclusive result has been reached
    self.maxFixations = (self.numStimuli * 1.5)  # TODO: Make this a configurable param
    self.numFixations = 0
    self.firstSaccadeLatency = None
    
    # * Response output
    self.rpcClient = None
    try:
      self.rpcClient = rpc.Client(port=ImageServer.default_port)
      return  # skip checking, otherwise will block
      testResult = self.rpcClient.call('rpc.list')
      if testResult == None:
        self.logger.warning("Did not get RPC result (no remote keyboard)")
        self.rpcClient.close()
        self.rpcClient = None
      else:
        self.logger.info("Remote keyboard connected (at least an RPC server is there)")
    except Exception as e:
      self.logger.error("Error initializing RPC client (no remote keyboard): {}".format(e))
  
  def update(self):
    # TODO: If vision is fixated, hold, match shape patterns (templates) with fixation region
    #       If it's a match for target, respond 'y' and return False to end
    #       If not, increment distractor count; if numDistractorsSeen >= numStimuli, respond 'n' and return False to end
    # NOTE(4/8): Above logic has mostly been implemented here
    visState = self.visSys.getBuffer('state')
    if visState != VisualSystem.State.FIXATE:
      self.newFixation = True  # if system is not currently fixated, next one must be a fresh one
      self.processFixation = True  # similarly reset this (NOTE: this is somewhat inefficient - setting flags repeatedly)
      if visState == VisualSystem.State.SACCADE and self.firstSaccadeLatency is None:
        self.firstSaccadeLatency = self.context.timeNow
        self.logger.info("First saccade at: {}".format(self.firstSaccadeLatency))
      return True
    
    #cv2.imshow("Fixation", self.visSys.getFixatedImage(self.featureChannel))  # [debug]
    if self.newFixation:
      self.visSys.setBuffer('intent', 'hold')  # ask visual system to hold fixation till we've processed it (subject to a max hold time)
      self.newFixation = False  # pass hold intent only once
    
    if not self.processFixation:  # to prevent repeated processing of the same fixation location (double-counting)
      return True
    
    # Get foveal/fixated image area: Use key/channel = 'BGR' for full-color matching, 'V' for intensity only, 'H' for hue only (colored bars), etc.
    #imageFovea = self.visSys.getFovealImage(self.featureChannel)
    imageFovea = self.visSys.getFixatedImage(self.featureChannel)  # TODO: rename imageFovea -> imageInFocus?
    #imageFovea = cv2.cvtColor(imageFovea, cv2.COLOR_BGR2GRAY)  # convert BGR to grayscale, if required
    
    if self.detect_fixation:
      matchF, minMatchF, maxMatchF, minMatchLocF, maxMatchLocF = self.getMatch(imageFovea, self.fixcross_pattern)
      self.logger.info("Match (F): min: {} at {}, max: {} at {}".format(minMatchF, minMatchLocF, maxMatchF, maxMatchLocF))
      if minMatchF <= 0.001:
        if hypot(*self.visSys.getBuffer('location')) < 20.0 and \
            hypot(*self.visSys.ocularMotionSystem.getFocusOffset()) < 20.0:  # a fixation cross in the center!
          self.logger.info("I think this is a fixation cross")
          # TODO: Reset vision system and be ready (use state to prevent double action)
    
    matchO, minMatchO, maxMatchO, minMatchLocO, maxMatchLocO = self.getMatch(imageFovea, self.o_pattern)
    #self.logger.info("Match (O): min: {} at {}, max: {} at {}".format(minMatchO, minMatchLocO, maxMatchO, maxMatchLocO))
    
    matchQ, minMatchQ, maxMatchQ, minMatchLocQ, maxMatchLocQ = self.getMatch(imageFovea, self.q_pattern)
    #self.logger.info("Match (Q): min: {} at {}, max: {} at {}".format(minMatchQ, minMatchLocQ, maxMatchQ, maxMatchLocQ))
    
    # Method: SQDIFF
    matchValueO = minMatchO
    matchValueQ = minMatchQ
    if minMatchO < minMatchQ:
      bestMatch = 'O'
      bestMatchValue = minMatchO
      bestMatchLoc = minMatchLocO
    else:
      bestMatch = 'Q'
      bestMatchValue = minMatchQ
      bestMatchLoc = minMatchLocQ
    
    '''
    # Method: CCORR, CCOEFF
    if maxMatchO > maxMatchQ:
      bestMatch = 'O'
      bestMatchValue = maxMatchO
      bestMatchLoc = maxMatchLocO
    else:
      bestMatch = 'Q'
      bestMatchValue = maxMatchQ
      bestMatchLoc = maxMatchLocQ
    '''
    
    self.logger.info("Best match: {}: {:.3f} at {} (O: [{:.3f}, {:.3f}] vs Q: [{:.3f}, {:.3f}])".format(bestMatch, bestMatchValue, bestMatchLoc, minMatchO, maxMatchO, minMatchQ, maxMatchQ))
    if bestMatchValue <= self.max_match_sqdiff and abs(matchValueO - matchValueQ) >= self.min_confidence_sqdiff:
      # We have a good match
      self.numFixations += 1  # can be treated as a valid fixation
      self.processFixation = False  # make sure we don't process this fixated location again
      if self.context.options.gui:  # if GUI, mark matched region
        cv2.rectangle(imageFovea, bestMatchLoc, (bestMatchLoc[0] + self.o_pattern.shape[1], bestMatchLoc[1] + self.o_pattern.shape[0]), int(255 * (1.0 - bestMatchValue)), 1)  # SQDIFF
      
      # Now decide what to do based on match
      if bestMatch == self.target:  # found the target!
        self.logger.info("Target found!")
        self.respond('y')
        return False  # end trial (TODO: start new trial?)
      else:  # nope, this ain't the target
        self.numDistractorsSeen += 1
        self.logger.info("Distractor (num seen: {})".format(self.numDistractorsSeen))
        if self.numDistractorsSeen >= self.numStimuli or self.numFixations >= self.maxFixations:  # all stimuli were (probably) distractors, no target
          self.respond('n')
          return False  # end trial (TODO: start new trial?)
        else:
          self.visSys.setBuffer('intent', 'release')  # let visual system inhibit and move on
    
    if self.context.options.gui:
      cv2.imshow("Fovea", imageFovea)
      #cv2.imshow("Match (O)", matchO)
      #cv2.imshow("Match (Q)", matchQ)
    
    return True
  
  def getMatch(self, img, templ):
    match = cv2.matchTemplate(img, templ, cv2.TM_SQDIFF)
    match /= (match.shape[1] * match.shape[0] * 255.0 * 255.0)  # normalize it
    #match = np.abs(match)
    #val, match = cv2.threshold(match, 0.01, 0, cv2.THRESH_TOZERO)
    minMatch, maxMatch, minMatchLoc, maxMatchLoc = cv2.minMaxLoc(match)
    #match8 = cv2.normalize(match, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    match8 = np.uint8(match * 255.0)  # for display (better return None if no GUI)
    return match8, minMatch, maxMatch, minMatchLoc, maxMatchLoc
  
  def respond(self, key):
    self.logger.info("Response: {}, time: {}, numFixations: {}, firstSaccadeLatency: {}, input: {}".format(key, self.context.timeNow, self.numFixations, self.firstSaccadeLatency, os.path.basename(self.context.options.input_source) if (self.context.isImage or self.context.isVideo) else 'live/rpc'))
    if not self.rpcClient is None:
      try:
        self.rpcClient.call('Keyboard.keyPress', params={'symbol': key})
      except Exception as e:
        self.logger.error("Error sending response key (will not retry): {}".format(e))
        self.rpcClient = None


if __name__ == "__main__":
  #VisualSearchAgent().run()
  ZelinksyFinder().run()
