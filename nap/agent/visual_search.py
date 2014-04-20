"""Visual search using features from visual system and ocular motion."""

import os
from math import hypot
import logging
import argparse
import numpy as np
import cv2
from collections import namedtuple, OrderedDict

from lumos.context import Context
from lumos.input import InputRunner
from lumos import rpc
from lumos.net import ImageServer
from lumos.util import Enum

from ..vision.visual_system import VisualSystem, Finst, VisionManager, FeatureManager, default_feature_weight, default_feature_weight_rest, default_window_flags


class VisualSearchAgent(object):
  """A simple visual search agent that scans input stream for locations with desired features."""
  
  image_size = (512, 512)  #VisualSystem.default_image_size  # size of retina to project on
  screen_background = np.uint8([0, 0, 0])  #VisionManager.default_screen_background
  
  def __init__(self):
    # * Create application context, passing in custom arguments, and get a logger
    argParser = argparse.ArgumentParser(add_help=False)
    argParser.add_argument('--features', type=str, default=None, help="features to look for, comma separated")
    self.context = Context.createInstance(description=self.__class__.__name__, parent_argparsers=[argParser])
    self.logger = logging.getLogger(self.__class__.__name__)
    
    # * Parse arguments
    self.features = self.context.options.features.split(',') if (hasattr(self.context.options, 'features') and self.context.options.features is not None) else []
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
  
  default_fixation_symbol = '+'  # just a name, actual pattern loaded from file
  default_target = 'Q'  # 'O' or 'Q' depending on condition
  default_distractor = 'O'  # commonly-used distractor
  default_distractors = [default_distractor]  # one or more types of distractors
  default_num_stimuli = 5  # 5 or 17 depending on condition
  default_num_trials = 4  # total no. of trials we should run for
  
  #pattern_size = (48, 48)  # size of pattern image [unused]
  #o_radius = 16  # radius of O pattern [unused]
  
  # NOTE: Pathnames relative to root of repository (add more to enable matching with other shapes/patterns)
  pattern_files = dict()
  pattern_files[default_fixation_symbol] = "res/data/visual-search/zelinsky-patterns/fixcross-pattern.png"
  pattern_files[default_target] = "res/data/visual-search/zelinsky-patterns/q-pattern.png"
  pattern_files[default_distractor] = "res/data/visual-search/zelinsky-patterns/o-pattern.png"
  
  # TODO: These need to be updated according to fovea and pattern size
  max_match_sqdiff = 0.01  # max value of SQDIFF matching to be counted as a valid match
  min_confidence_sqdiff = 0.008  # min desired difference between activations for target and distractor (to avoid jumping to conclusion when there is confusion)
  
  State = Enum(('NONE', 'PRE_TRIAL', 'TRIAL', 'POST_TRIAL'))  # explicit tracking of experiment state (TODO: implement actual usage)
  
  def __init__(self, fixationSymbol=default_fixation_symbol, target=default_target, distractors=default_distractors, numStimuli=default_num_stimuli, numTrials=default_num_trials, featureChannel='V'):
    VisualSystem.num_finsts = numStimuli  # override FINST size (TODO: make FINSTs fade out, design multi-scale FINSTs to cover larger areas/clusters for a better model)
    VisualSystem.finst_decay_enabled = True
    Finst.half_life = numStimuli * 4.0
    Finst.default_radius = 64
    VisualSearchAgent.__init__(self)
    self.fixationSymbol = fixationSymbol
    self.target = target
    self.distractors = distractors  # None (not supported yet) could mean everything else is a distractor (other than fixation symbol)
    self.featureChannel = featureChannel
    self.numStimuli = numStimuli
    self.numTrials = numTrials  # NOTE(04/19/2014): currently unused
    
    # * Configure visual system as needed (shorter times for fast completion goal, may result in some inaccuracy)
    self.visSys.max_free_duration = 0.25
    self.visSys.max_fixation_duration = 0.5  # we don't need this to be high as we are using the hold-release pattern
    self.visSys.max_hold_duration = 2.0
    self.visSys.min_good_salience = 0.2  # this task is generally very low-salience
    #self.visSys.min_saccade_salience = 0.1
    self.visSys.ocularMotionSystem.enableEventLogging("ocular-events_{}".format(self.target))
    
    # * Initialize pattern matchers (load only those patterns that are needed; TODO: use different flags for each pattern? color/grayscale)
    self.patternMatchers = OrderedDict()
    missingPatterns = [] 
    if self.fixationSymbol in self.pattern_files:
      self.patternMatchers[self.fixationSymbol] = PatternMatcher(self.fixationSymbol, self.pattern_files[self.fixationSymbol], flags=0)  # pass flags=0 for grayscale
    else:
      missingPatterns.append(self.fixationSymbol)
    if self.target in self.pattern_files:
      self.patternMatchers[self.target] = PatternMatcher(self.target, self.pattern_files[self.target], flags=0)
    else:
      missingPatterns.append(self.target)
    if self.distractors is not None:
      for distractor in self.distractors:
        if distractor in self.pattern_files:
          self.patternMatchers[distractor] = PatternMatcher(distractor, self.pattern_files[distractor], flags=0)
        else:
          missingPatterns.append(self.fixationSymbol)
    if missingPatterns:
      self.logger.error("Patterns missing (matching may not be correct): {}".format(missingPatterns))
    
    # * Initialize matching-related objects
    # TODO Turn this into a state machine + some vars
    self.numDistractorsSeen = 0  # if numDistractorsSeen >= numStimuli, no target is present
    self.newFixation = False
    self.processFixation = False  # used to prevent repeatedly processing a fixation period once a conclusive result has been reached
    self.maxFixations = (self.numStimuli * 1.5)  # TODO: Make this a configurable param
    self.numFixations = 0
    self.firstSaccadeLatency = None
    self.imageInFocus = self.visSys.getFixatedImage(self.featureChannel)
    self.targetFound = None  # flag, mainly for visualization
    
    # * Trial awareness
    self.trialCount = 0
    self.trialStarted = None  # None means we haven't seen fixation point yet, otherwise time trial began
    
    # * Response output
    self.rpcClient = None
    try:
      self.rpcClient = rpc.Client(port=ImageServer.default_port, timeout=1000)
      self.logger.info("Checking for remote keyboard via RPC")
      testResult = self.rpcClient.call('rpc.list')
      if testResult == None:
        self.logger.warning("Did not get RPC result (no remote keyboard)")
        self.rpcClient.close()
        self.rpcClient = None
      else:
        self.logger.info("Remote keyboard connected (at least an RPC server is there)")
    except Exception as e:
      self.logger.error("Error initializing RPC client (no remote keyboard): {}".format(e))
    
    if self.context.options.gui:
      self.isOutputInverted = True  # True = black on white, False = white on black
      self.winName = "Zelinsky visual search agent"  # primary output window name
      self.imageOut = np.full((self.visSys.imageSize[1] * 2, self.visSys.imageSize[0] * 2, 3), 0.0, dtype=self.visSys.imageTypeInt)  # 3-channel composite output image
      cv2.namedWindow(self.winName, flags=default_window_flags)
  
  def update(self):
    # TODO: If vision is fixated, hold, match shape patterns (templates) with fixation region
    #       If it's a match for target, respond 'y' and end current trial
    #       If not (and matches a distractor), increment distractor count; if numDistractorsSeen >= numStimuli, respond 'n' and end trial
    # NOTE(4/8): Above logic has mostly been implemented here, save for repeated trial handling
    visState = self.visSys.getBuffer('state')
    if visState != VisualSystem.State.FIXATE:
      self.newFixation = True  # if system is not currently fixated, next one must be a fresh one
      self.processFixation = True  # similarly reset this (NOTE: this is somewhat inefficient - setting flags repeatedly)
      if visState == VisualSystem.State.SACCADE and self.firstSaccadeLatency is None:
        self.firstSaccadeLatency = self.context.timeNow
        self.logger.info("First saccade at: {}".format(self.firstSaccadeLatency))
      if self.context.options.gui:
        self.visualize()  # duplicate call (TODO: consolidate)
      return True
    
    if self.newFixation:
      self.visSys.setBuffer('intent', 'hold')  # ask visual system to hold fixation till we've processed it (subject to a max hold time)
      self.targetFound = None
      self.newFixation = False  # pass hold intent only once
    
    if not self.processFixation:  # to prevent repeated processing of the same fixation location (double-counting)
      if self.context.options.gui:
        self.visualize()  # duplicate call (TODO: consolidate)
      return True
    
    # Get foveal/fixated image area (image-in-focus): Use key/channel = 'BGR' for full-color matching, 'V' for intensity only, 'H' for hue only (colored bars), etc.
    #self.imageInFocus = self.visSys.getFovealImage(self.featureChannel)
    self.imageInFocus = self.visSys.getFixatedImage(self.featureChannel)
    #self.imageInFocus = cv2.cvtColor(self.imageInFocus, cv2.COLOR_BGR2GRAY)  # convert BGR to grayscale, if required
    #cv2.imshow("Focus", self.imageInFocus)  # [debug]
    if self.imageInFocus.shape[0] < self.visSys.foveaSize[0] or self.imageInFocus.shape[1] < self.visSys.foveaSize[1]:
      if self.context.options.gui:
        self.visualize()  # duplicate call (TODO: consolidate)
      return True  # fixated on a weird, incomplete spot
    
    # Compute matches and best match in a loop
    matches = [matcher.match(self.imageInFocus) for matcher in self.patternMatchers.itervalues()]  # combined matching
    self.logger.info("Matches: %s", ", ".join("{}: {:.3f} at {}".format(match.matcher.name, match.value, match.location) for match in matches))  # combined reporting
    
    #matchO = self.patternMatchers['O'].match(self.imageInFocus)  # individual matching
    #self.logger.info("Match (O): value: {:.3f} at {}".format(matchO.value, matchO.location))  # individual reporting
    
    bestMatch = min(matches, key=lambda match: match.value)  # TODO: sort to find difference between two best ones: (bestMatch[1].value - bestMatch[0].value) >= self.min_confidence_sqdiff
    if bestMatch.value <= self.max_match_sqdiff:  # for methods CCORR, CCOEFF, use: bestMatch.value >= self.min_match_XXX
      # We have a good match
      self.logger.info("Good match: {}: {:.3f} at {}".format(bestMatch.matcher.name, bestMatch.value, bestMatch.location))
      self.numFixations += 1  # can be treated as a valid fixation
      self.processFixation = False  # make sure we don't process this fixated location again
      if self.context.options.gui:  # if GUI, mark matched region (NOTE: this modifies foveal image!)
        cv2.rectangle(self.imageInFocus, bestMatch.location, (bestMatch.location[0] + bestMatch.matcher.pattern.shape[1], bestMatch.location[1] + bestMatch.matcher.pattern.shape[0]), int(255.0 * (1.0 - bestMatch.value)))
        cv2.putText(self.imageInFocus, str(bestMatch.matcher.name), (bestMatch.location[0] + 2, bestMatch.location[1] + 10), cv2.FONT_HERSHEY_PLAIN, 0.67, 200)
      
      # Now decide what to do based on match
      symbol = bestMatch.matcher.name  # bestMatch.matcher.name contains the symbol specified when creating the corresponding pattern
      if symbol == self.fixationSymbol:  # NOTE: system doesn't always catch fixation symbol, so don't rely on this - keep playing
        if bestMatch.value <= 0.001 and \
            hypot(*self.visSys.getBuffer('location')) < 20.0 and \
            hypot(*self.visSys.ocularMotionSystem.getFocusOffset()) < 20.0:  # a fixation cross, and approx. in the center!
          self.logger.info("Fixation symbol!")
          # TODO: Prepare vision system to reset (hold till cross disappears? don't add FINST?) and be ready (use state to prevent double action)
          if self.trialStarted is None:
            self.logger.info("Trial {}: Fixation symbol seen; assuming trial starts now".format(self.trialCount))
            self.trialStarted = self.context.timeNow
        self.visSys.setBuffer('intent', 'release')  # let visual system inhibit and move on
      elif symbol == self.target:  # found the target!
        self.respond('y')  # response is primary!
        self.logger.info("Target found!")
        self.targetFound = True
        self.nextTrial()
        #return True  # ready for new trial (will happen anyways, might as well show foveal image)
        #return False  # end trial and stop this run
      elif self.distractors is None or symbol in self.distractors:
        self.numDistractorsSeen += 1
        self.logger.info("Distractor {}".format(self.numDistractorsSeen))
        if self.numDistractorsSeen >= self.numStimuli or self.numFixations >= self.maxFixations:  # all stimuli were (probably) distractors, no target
          self.respond('n')
          self.targetFound = False
          self.nextTrial()
          #return True  # ready for new trial (will happen anyways, might as well show foveal image)
          #return False  # end trial and stop this run
        else:
          self.visSys.setBuffer('intent', 'release')  # let visual system inhibit and move on
      else:
        self.logger.warning("I don't know what that is; ignoring fixation")
        self.visSys.setBuffer('intent', 'release')  # let visual system inhibit and move on
    
    # Visualize system operation
    if self.context.options.gui:
      self.visualize()
      #if self.context.options.debug:
      #  for match in matches:
      #    cv2.imshow("Match ({})".format(match.matcher.name), match.result)
    
    return True
  
  def respond(self, key):
    self.logger.info("Response: {}, time: {}, numFixations: {}, firstSaccadeLatency: {}, input: {}".format(key, self.context.timeNow, self.numFixations, self.firstSaccadeLatency, os.path.basename(self.context.options.input_source) if (self.context.isImage or self.context.isVideo) else 'live/rpc'))
    if not self.rpcClient is None:
      try:
        self.rpcClient.call('Keyboard.keyPress', params={'symbol': key})
      except Exception as e:
        self.logger.error("Error sending response key (will not retry): {}".format(e))
        self.rpcClient = None
  
  def nextTrial(self):
    self.trialCount += 1
    self.trialStarted = None
    
    self.numDistractorsSeen = 0
    self.newFixation = False
    self.processFixation = False
    self.numFixations = 0
    self.firstSaccadeLatency = None
    
    self.visSys.setBuffer('intent', 'reset')
  
  def visualize(self):
    # Combine individual outputs into giant composite image
    self.imageOut[0:self.visMan.imageSize[1], 0:self.visMan.imageSize[0]] = self.visMan.image  # input
    self.imageOut[0:self.visSys.imageSize[1], (self.imageOut.shape[1] - self.visSys.imageSize[0]):self.imageOut.shape[1]] = self.visSys.images['BGR']  # retina
    self.imageOut[0:self.imageInFocus.shape[0], (self.imageOut.shape[1] - self.imageInFocus.shape[1]):self.imageOut.shape[1]] = cv2.cvtColor(self.imageInFocus, cv2.COLOR_GRAY2BGR)  # foveal image, inset top-right
    self.imageOut[(self.imageOut.shape[0] - self.visSys.imageSize[1]):self.imageOut.shape[0], 0:self.visSys.imageSize[0]].fill(0)
    if self.context.options.debug:
      #self.imageOut[(self.imageOut.shape[0] - self.visSys.imageSize[1]):self.imageOut.shape[0], 0:self.visSys.imageSize[0]] = cv2.cvtColor(self.visSys.imageSalienceOutCombined, cv2.COLOR_GRAY2BGR)  # combined salience (neuron firings)
      self.imageOut[(self.imageOut.shape[0] - self.visSys.imageSize[1]):self.imageOut.shape[0], 0:self.visSys.imageSize[0], 1] = cv2.convertScaleAbs(self.visSys.imageSalienceOutCombined, alpha=5, beta=0)  # combined salience (neuron firings): Magenta = high
    #self.imageOut[(self.imageOut.shape[0] - self.visSys.imageSize[1]):self.imageOut.shape[0], (self.imageOut.shape[1] - self.visSys.imageSize[0]):self.imageOut.shape[1]] = cv2.cvtColor(self.visSys.imageOut, cv2.COLOR_GRAY2BGR)  # VisualSystem output salience and labels/marks
    if self.isOutputInverted:
      self.imageOut = 255 - self.imageOut
    # Colored visualizations, post-inversion
    self.imageOut[(self.imageOut.shape[0] - self.visSys.imageSize[1]):self.imageOut.shape[0], (self.imageOut.shape[1] - self.visSys.imageSize[0]):self.imageOut.shape[1], 0] = 255 - self.visSys.imageOut  # VisualSystem output salience and labels/marks: Blue = low
    self.imageOut[(self.imageOut.shape[0] - self.visSys.imageSize[1]):self.imageOut.shape[0], (self.imageOut.shape[1] - self.visSys.imageSize[0]):self.imageOut.shape[1], 1] = cv2.convertScaleAbs(self.visSys.imageOut, alpha=4, beta=0)  # VisualSystem output salience and labels/marks: Green = high
    self.imageOut[(self.imageOut.shape[0] - self.visSys.imageSize[1]):self.imageOut.shape[0], (self.imageOut.shape[1] - self.visSys.imageSize[0]):self.imageOut.shape[1], 2] = cv2.convertScaleAbs(self.visSys.imageOut, alpha=4, beta=0)  # VisualSystem output salience and labels/marks: Red = high (combined with Green, Yellow = high)
    #cv2.convertScaleAbs(self.imageOut, dst=self.imageOut, alpha=0, beta=255)
    #imageOutGray = cv2.cvtColor(self.imageOut, cv2.COLOR_BGR2GRAY)  # converting back to grayscale, very inefficient
    #cv2.equalizeHist(imageOutGray, dst=imageOutGray)
    
    # Draw frames, labels, marks and show
    cv2.rectangle(self.imageOut, (0, 0), (self.imageOut.shape[1] - 1, self.imageOut.shape[0] - 1), (128, 128, 128), 2)  # outer border
    cv2.line(self.imageOut, (0, self.imageOut.shape[0] / 2), (self.imageOut.shape[1], self.imageOut.shape[0] / 2), (128, 128, 128), 2)  # inner border, horizontal
    cv2.line(self.imageOut, (self.imageOut.shape[1] / 2, 0), (self.imageOut.shape[1] / 2, self.imageOut.shape[0]), (128, 128, 128), 2)  # inner border, vertical
    cv2.rectangle(self.imageOut, (self.imageOut.shape[1] - self.imageInFocus.shape[1], 0), (self.imageOut.shape[1] - 1, self.imageInFocus.shape[0] - 1), (128, 128, 128), 2)  # inset border, top-right
    self.labelImage(self.imageOut, "Input", (20, self.imageOut.shape[0] / 2 - 20))
    self.labelImage(self.imageOut, "Retina", (self.imageOut.shape[1] / 2 + 20, self.imageOut.shape[0] / 2 - 20))
    self.labelImage(self.imageOut, "Focus", (self.imageOut.shape[1] - self.imageInFocus.shape[1] + 12, self.imageInFocus.shape[0] + 24), color=(128, 128, 128), bgColor=None)
    self.labelImage(self.imageOut, "Neuron activity", (20, self.imageOut.shape[0] - 20))
    self.labelImage(self.imageOut, "Output visualization", (self.imageOut.shape[1] / 2 + 20, self.imageOut.shape[0] - 20))
    focusRectColor = (64, 64, 64)  # gray, default
    if self.targetFound is not None:
      focusRectColor = (16, 128, 16) if self.targetFound else (16, 16, 128)  # green if targetFound else red
      self.labelImage(self.imageOut, "Y" if self.targetFound else "N", (self.imageOut.shape[1] / 2 + self.visSys.fixationSlice[1].start + 6, self.imageOut.shape[0] / 2 + self.visSys.fixationSlice[0].stop - 6), color=focusRectColor, bgColor=None)
    cv2.rectangle(self.imageOut, (self.imageOut.shape[1] / 2 + self.visSys.fixationSlice[1].start, self.imageOut.shape[0] / 2 + self.visSys.fixationSlice[0].start), (self.imageOut.shape[1] / 2 + self.visSys.fixationSlice[1].stop, self.imageOut.shape[0] / 2 + self.visSys.fixationSlice[0].stop), focusRectColor, 2)  # focus rect in output image
    cv2.imshow(self.winName, self.imageOut)
    #cv2.imshow(self.winName, imageOutGray)
  
  def labelImage(self, img, text, org, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(200, 200, 200), thickness=2, bgColor=(32, 32, 32)):
    """Wrapper around cv2.putText to add a background/box."""
    # TODO: Make this a util function
    if bgColor is not None:
      textSize, baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
      cv2.rectangle(img, (org[0] - baseline, org[1] + baseline + 2), (org[0] + textSize[0] + baseline, org[1] - textSize[1] - baseline), bgColor, -1)
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness)


PatternMatch = namedtuple('PatternMatch', ['value', 'location', 'result', 'matcher'])


class PatternMatcher(object):
  """Helper class to perform simple pattern matching - a higher-level visual function not modeled in the framework."""
  
  default_method = cv2.TM_SQDIFF
  
  def __init__(self, name, pattern_file, flags=1, method=default_method):
    """Load a pattern from file and specify a method for matching. Param flags is directly passed on to cv2.imread(): 1 = auto, 0 = grayscale."""
    self.name = name
    self.pattern_file = pattern_file
    self.method = method
    self.pattern = cv2.imread(self.pattern_file, flags)  # flags=0 for grayscale
    #self.pattern = cv2.blur(self.pattern, (3, 3))  # not good for precise stimuli discrimination, like between O and Q-like
    #cv2.imshow("Pattern ({})".format(self.name), self.pattern) ## [debug]
  
  def match(self, image):
    """Match pattern with image, return a 3-tuple: (<uint8 result map>, <best match value>, <best match location>)."""
    result = cv2.matchTemplate(image, self.pattern, self.method)
    result /= (result.shape[1] * result.shape[0] * 255.0 * 255.0)  # normalize result, dividing by sum of max possible differences
    #result = np.abs(result)
    #val, result = cv2.threshold(result, 0.01, 0, cv2.THRESH_TOZERO)
    minMatch, maxMatch, minMatchLoc, maxMatchLoc = cv2.minMaxLoc(result)
    
    #result_uint8 = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)  # normalize (issue is variable scale)
    result_uint8 = np.uint8(result * 255.0)  # scale, for display (better avoid and return None if no GUI)
    
    #return result_uint8, minMatch, maxMatch, minMatchLoc, maxMatchLoc  # too many returns, generalize to *best* match value and loc
    if self.method == cv2.TM_SQDIFF or self.method == cv2.TM_SQDIFF_NORMED:
      return PatternMatch(value=minMatch, location=minMatchLoc, result=result_uint8, matcher=self)
    else:  # TM_CCORR or TM_CCOEFF
      return PatternMatch(value=maxMatch, location=maxMatchLoc, result=result_uint8, matcher=self)


if __name__ == "__main__":
  argParser = argparse.ArgumentParser(add_help=False)
  argParser.add_argument('--zelinsky', action='store_true', help="run a Zelinsky search agent")
  argParser.add_argument('--target', default='Q', choices=('Q', 'O'), help='target symbol (Q or O)')
  argParser.add_argument('--size', dest='num_stimuli', type=int, default=5, help='display size (no. of stimuli) to expect')
  argParser.add_argument('--features', type=str, default=None, help="features to look for, comma separated")  # duplicated for VisualSearchAgent (TODO: Find a better way to unify args, parsers)
  context = Context.createInstance(description="Zelinsky search agent", parent_argparsers=[argParser])
  if context.options.zelinsky:
    if context.options.features is None:
      context.options.features = 'OFF:1.0'  # Zelinsky-specific
    ZelinksyFinder(target=context.options.target, distractors=('O' if context.options.target == 'Q' else 'Q'), numStimuli=context.options.num_stimuli).run()
  else:
    VisualSearchAgent().run()
  
  # Some example invocations
  #ZelinksyFinder(target='Q', distractors=['O'], numStimuli= 5).run()  # target: 'Q', distractor: 'O'; size:  5 [default]
  #ZelinksyFinder(target='Q', distractors=['O'], numStimuli=17).run()  # target: 'Q', distractor: 'O'; size: 17
  #ZelinksyFinder(target='O', distractors=['Q'], numStimuli= 5).run()  # target: 'O', distractor: 'Q'; size:  5
  #ZelinksyFinder(target='O', distractors=['Q'], numStimuli=17).run()  # target: 'O', distractor: 'Q'; size: 17
