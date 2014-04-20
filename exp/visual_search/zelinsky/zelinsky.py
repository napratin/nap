#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy2 Experiment Builder (v1.78.01), Tue 08 Apr 2014 04:37:15 AM EDT
If you publish work using this script please cite the relevant PsychoPy publications
  Peirce, JW (2007) PsychoPy - Psychophysics software in Python. Journal of Neuroscience Methods, 162(1-2), 8-13.
  Peirce, JW (2009) Generating stimuli for neuroscience using PsychoPy. Frontiers in Neuroinformatics, 2:10. doi: 10.3389/neuro.11.010.2008
"""

from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
from psychopy import visual, core, data, event, logging, sound, gui
from psychopy.constants import *  # things like STARTED, FINISHED
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import sin, cos, tan, log, log10, pi, average, sqrt, std, deg2rad, rad2deg, linspace, asarray
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions

# Experiment: zelinksy - a small set of visual search tasks
# Based on Zelinksy et al. 1995, 1997
import argparse
from math import hypot, atan2
from collections import OrderedDict

# Trial parameters
argParser = argparse.ArgumentParser(description="Zelinsky search task")
argParser.add_argument('--target', default='Q', choices=('Q', 'O'), help='target symbol (Q or O)')
argParser.add_argument('--size', dest='num_stimuli', type=int, default=5, help='display size (no. of stimuli)')
argParser.add_argument('--reps', dest='num_reps', type=int, default=2, help='no. of repetitions (#trials = num_reps * factor levels)')
argParser.add_argument('--hide_text', action="store_true", help="hide instruction text? (useful for automated play)")
options = argParser.parse_args()

# * Frozen factors (remaining factor: target presence)
target = options.target  # 'Q' or 'O'
num_stimuli = options.num_stimuli  # 5 or 17
# * Repetitions
num_reps = options.num_reps  # 128 (NOTE: num_trials = num_reps * all factor levels)
# * Other, dependent
distractor = 'O' if target == 'Q' else 'Q'  # 'O' or 'Q', determined by target
random_seed = num_stimuli + ord(target)  # used to set numpy RNG's seed
np.random.seed(random_seed)  # TODO: ensure no one else re-seeds this later
fixationCrossDuration = 4.0  # secs.; how long the fixation cross should stay
save_images = False  # TODO: complete this functionality, then enable to save snapshots
if options.hide_text:
    instructions = u""  # NOTE: pre-trial screen still waits for keypress
else:
    instructions = u"Experiment: Find the target\n\nAt the beginning of each trial, look at the cross.\nA set of shapes will then appear, with a possible target (odd one out).\nPress 'y' if you saw the target, 'n' if you didn't.\nTry to respond as quickly as you can.\n\n[Press any key to continue]"

# [Serve] Imports and initialization (enable logging for debugging/testing only)
from nap.util.net import ImageServerWindow
ImageServerWindow.flipVertical = True  # psychopy uses (0, 0) to mean bottom-left
visual.Window = ImageServerWindow  # override psychopy Window with a subclass

# [Play] Accept keystrokes remotely
from nap.util.net import RemoteKeyboard
# NOTE This should expose a Keyboard.keyPress RPC call, activated when RPC server is started.
#      E.g.: When ImageServerWindow is initialized.

# [Log] Record experiment events to file
import time
from lumos.net import EventLogger
logEvents = True
if logEvents:
    eventTag = "TRIAL"
    eventFilename = "logs/trial-events_{}_{}_{}.log".format(target, num_stimuli, time.strftime(EventLogger.timestamp_format, time.localtime(time.time())))
    eventLogger = EventLogger(eventFilename, rpc_export=False, start_server=False)  # no need to expose event logger, this is for local logging only (TODO: design a logger proxy and then enable this to provide a common logging target)

# Store info about the experiment session
expName = u'zelinsky'  # from the Builder filename that created this script
expInfo = {u'session': u'000', u'participant': u'0'}
expInfo['expName'] = expName
expInfo['date'] = data.getDateStr('%Y-%m-%d_%H-%M-%S')  # add a simple timestamp
expInfo['target'] = target
expInfo['num_stimuli'] = num_stimuli

# Setup files for saving
if not os.path.isdir('data'):
    os.makedirs('data')  # if this fails (e.g. permissions) we will get error
filename = 'data' + os.path.sep + '%s_%s' %(expInfo['participant'], expInfo['date'])
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath=None,
    savePickle=True, saveWideText=True,
    dataFileName=filename)

# Setup the Window
win = visual.Window(size=[512, 512], fullscr=False, screen=0, allowGUI=True, allowStencil=False,
    monitor=u'testMonitor', color=[-1,-1,-1], colorSpace=u'rgb', units=u'deg')
# store frame rate of monitor if we can measure it successfully
expInfo['frameRate']=win._getActualFrameRate()
if expInfo['frameRate']!=None:
    frameDur = 1.0/round(expInfo['frameRate'])
else:
    frameDur = 1.0/60.0 # couldn't get a reliable measure so guess

# Initialize components for Routine "instr"
instrClock = core.Clock()
instrText = visual.TextStim(win=win, ori=0, name='instrText',
    text=instructions,    font=u'Arial',
    pos=[0, 0], height=0.5, wrapWidth=None,
    color=u'white', colorSpace=u'rgb', opacity=1,
    depth=0.0)

# Initialize components for Routine "blank"
blankClock = core.Clock()
blankISI = core.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='blankISI')

# Initialize components for Routine "trial"
trialClock = core.Clock()
fixationCross = visual.TextStim(win=win, ori=0, name='fixationCross',
    text='+',    font='Arial',
    pos=[0, 0], height=1, wrapWidth=None,
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0)


# Function defs
"""
Convert from polar (r,w) to rectangular (x,y)
    x = r cos(w)
    y = r sin(w)
"""
def rect(r, w, deg=1):  # radian if deg=0; degree if deg=1
    if deg:
        w = pi * w / 180.0
    return r * cos(w), r * sin(w)


"""
Convert from rectangular (x,y) to polar (r,w)
    r = sqrt(x^2 + y^2)
    w = arctan(y/x) = [-\pi,\pi] = [-180,180]
"""
def polar(x, y, deg=1): # radian if deg=0; degree if deg=1
    if deg:
        return hypot(x, y), 180.0 * atan2(y, x) / pi
    else:
        return hypot(x, y), atan2(y, x)


# Define stimuli to be used as target and distractor, supporting functions
def getStimulus(symbol):
    stim = None
    if symbol == 'Q':
        stim = (visual.Polygon(win=win, name='Q_body',
                    edges = 30, size=[0.67, 0.67],
                    ori=0, pos=[0, 0],
                    lineWidth=2.0, lineColor=[1,1,1], lineColorSpace=u'rgb',
                    fillColor=[-1,-1,-1], fillColorSpace=u'rgb',
                    opacity=1, interpolate=True),
                visual.Line(win=win, name='Q_mark',
                    start=(-[0.67, 0][0]/2.0, 0), end=(+[0.67, 0][0]/2.0, 0),
                    ori=90, pos=[0, 0.33],
                    lineWidth=2.0, lineColor=[1,1,1], lineColorSpace=u'rgb',
                    fillColor=[-1,-1,-1], fillColorSpace=u'rgb',
                    opacity=1, interpolate=True),
                (0.0, 0.33))
    elif symbol == 'O':
        stim = visual.Polygon(win=win, name='O',
                    edges = 30, size=[0.67, 0.67],
                    ori=0, pos=[0, 0],
                    lineWidth=2.0, lineColor=[1,1,1], lineColorSpace=u'rgb',
                    fillColor=[-1,-1,-1], fillColorSpace=u'rgb',
                    opacity=1, interpolate=True)
    return stim


# TODO: Define CompositeStim class to encapsulate this functionality
def setStimulusPos(stim, pos, offset=(0.0, 0.0)):
    if isinstance(stim, tuple):  # this is a composite stimulus: (body, mark, markOffset)
        setStimulusPos(stim[0], pos)
        setStimulusPos(stim[1], pos, stim[2])
    else:
        stim.setPos([pos[0] + offset[0], pos[1] + offset[1]])


def startStimulus(stim):
    if isinstance(stim, tuple):  # this is a composite stimulus: (body, mark, markOffset)
        startStimulus(stim[0])
        startStimulus(stim[1])
    else:
        stim.tStart = t  # underestimates by a little under one frame
        stim.frameNStart = frameN  # exact frame index
        stim.setAutoDraw(True)


def stopStimulus(stim):
    if isinstance(stim, tuple):  # this is a composite stimulus: (body, mark, markOffset)
        stopStimulus(stim[0])
        stopStimulus(stim[1])
    else:
        stim.setAutoDraw(False)


targetStim = getStimulus(target)
distractorStims = [None] * num_stimuli
for i in xrange(num_stimuli):
    distractorStims[i] = getStimulus(distractor)

# Initialization, pre-computation
print "Zelinsky search task"
print "- target: {}, distractor: {}, num_stimuli: {}, num_reps: {}".format(target, distractor, num_stimuli, num_reps)
print "Pre-computed stimulus position data:-"
directions = np.linspace(0, 360, 16, endpoint=False)  # deg
eccentricities = np.linspace(3, 6, 1)  # deg
print "- directions:", directions
print "- eccentricities:", eccentricities

dir_by_ecc = OrderedDict()
dir_by_ecc[3] = (22.5 + np.linspace(0, 360, 4, endpoint=False)) % 360
dir_by_ecc[4] = (45 + np.linspace(0, 360, 8, endpoint=False)) % 360
dir_by_ecc[5] = (22.5 + np.linspace(0, 360, 8, endpoint=False)) % 360
dir_by_ecc[6] = (45 + np.linspace(0, 360, 4, endpoint=False)) % 360
print "- dir_by_ecc:\n  ", "\n  ".join("{}: {}".format(ecc, dirs) for ecc, dirs in dir_by_ecc.iteritems())

positions = []
for ecc, dirs in dir_by_ecc.iteritems():
    for dir in dirs:
        positions.append((ecc, dir))
print "- positions:", positions

dummy = visual.TextStim(win=win, ori=0, name='dummy',
    text=None,    font=u'Arial',
    pos=[0, 0], height=1, wrapWidth=None,
    color=u'white', colorSpace=u'rgb', opacity=1,
    depth=-2.0)

# Initialize components for Routine "blank"
blankClock = core.Clock()
blankISI = core.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='blankISI')

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

#------Prepare to start Routine "instr"-------
t = 0
instrClock.reset()  # clock 
frameN = -1
# update component parameters for each repeat
instrResponse = event.BuilderKeyResponse()  # create an object of type KeyResponse
instrResponse.status = NOT_STARTED
# keep track of which components have finished
instrComponents = []
instrComponents.append(instrText)
instrComponents.append(instrResponse)
for thisComponent in instrComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "instr"-------
continueRoutine = True
while continueRoutine:
    # get current time
    t = instrClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *instrText* updates
    if t >= 0.0 and instrText.status == NOT_STARTED:
        # keep track of start time/frame for later
        instrText.tStart = t  # underestimates by a little under one frame
        instrText.frameNStart = frameN  # exact frame index
        instrText.setAutoDraw(True)
    
    # *instrResponse* updates
    if t >= 0.0 and instrResponse.status == NOT_STARTED:
        # keep track of start time/frame for later
        instrResponse.tStart = t  # underestimates by a little under one frame
        instrResponse.frameNStart = frameN  # exact frame index
        instrResponse.status = STARTED
        # keyboard checking is just starting
        event.clearEvents()
    if instrResponse.status == STARTED:
        theseKeys = event.getKeys()
        if len(theseKeys) > 0:  # at least one key was pressed
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instrComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the [Esc] key)
    if event.getKeys(["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()
    else:  # this Routine was not non-slip safe so reset non-slip timer
        routineTimer.reset()

#-------Ending Routine "instr"-------
for thisComponent in instrComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)

#------Prepare to start Routine "blank"-------
t = 0
blankClock.reset()  # clock 
frameN = -1
routineTimer.add(0.500000)
# update component parameters for each repeat
# keep track of which components have finished
blankComponents = []
blankComponents.append(blankISI)
for thisComponent in blankComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "blank"-------
continueRoutine = True
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = blankClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    # *blankISI* period
    if t >= 0.0 and blankISI.status == NOT_STARTED:
        # keep track of start time/frame for later
        blankISI.tStart = t  # underestimates by a little under one frame
        blankISI.frameNStart = frameN  # exact frame index
        blankISI.start(0.5)
    elif blankISI.status == STARTED: #one frame should pass before updating params and completing
        blankISI.complete() #finish the static period
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in blankComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the [Esc] key)
    if event.getKeys(["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

#-------Ending Routine "blank"-------
for thisComponent in blankComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)

# set up handler to look after randomisation of conditions etc
block = data.TrialHandler(nReps=num_reps, method=u'random', 
    extraInfo=expInfo, originPath=None,
    trialList=data.importConditions(u'zelinsky_cond_present.csv'),
    seed=None, name='block')
thisExp.addLoop(block)  # add the loop to the experiment
thisBlock = block.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb=thisBlock.rgb)
if thisBlock != None:
    for paramName in thisBlock.keys():
        exec(paramName + '= thisBlock.' + paramName)

for thisBlock in block:
    currentLoop = block
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock.keys():
            exec(paramName + '= thisBlock.' + paramName)
    
    #------Prepare to start Routine "trial"-------
    t = 0
    trialClock.reset()  # clock 
    frameN = -1
    # update component parameters for each repeat
    #present = np.random.choice(2)  # 0 or 1 [now from conditions file]
    pos_indices = np.random.choice(len(positions), num_stimuli, replace=False)  # position indices
    print "Trial {}: present: {}, target: {}, num_stimuli: {} pos_indices: {}".format(block.thisN, present, target, num_stimuli, pos_indices)
    
    # generate stimuli (target and distractors)
    stims = []
    for i, pos_idx in enumerate(pos_indices):
        pos = rect(*positions[pos_idx])
        #print "i: {}, pos_idx: {}, pos (deg): {}, pos (rect): {}, stim: {}".format(i, pos_idx, positions[pos_idx], pos, (target if i < present else distractor))  # [debug]
        if i < present:  # use first position for target, if present
            setStimulusPos(targetStim, pos)
            stims.append(targetStim)
            #print "Target at:", pos  # [debug]
        else:  # all other position indices map nicely to distractors
            setStimulusPos(distractorStims[i], pos)
            stims.append(distractorStims[i])
            #print "Distractor at:", pos  # [debug]
    stims_status = NOT_STARTED  # to keep track of generated stimuli
    
    # save stimuli screens as images
    if save_images:
        screenshot = visual.BufferImageStim(win, stim=stims)
        # TODO: Save screenshot to file
    
    response = event.BuilderKeyResponse()  # create an object of type KeyResponse
    response.status = NOT_STARTED
    # keep track of which components have finished
    trialComponents = []
    trialComponents.append(fixationCross)
    trialComponents.append(dummy)
    trialComponents.append(response)
    for thisComponent in trialComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    #-------Start Routine "trial"-------
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = trialClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *fixationCross* updates
        if t >= 0.0 and fixationCross.status == NOT_STARTED:
            # keep track of start time/frame for later
            fixationCross.tStart = t  # underestimates by a little under one frame
            fixationCross.frameNStart = frameN  # exact frame index
            fixationCross.setAutoDraw(True)
        elif fixationCross.status == STARTED and t >= (0.0 + fixationCrossDuration):
            fixationCross.setAutoDraw(False)
        
        # *stims* (generated stimuli) updates
        if t >= fixationCrossDuration and stims_status == NOT_STARTED:
            # start generated stimuli
            for stim in stims:
                startStimulus(stim)
            stims_status = STARTED  # NOTE: have to manually set this
            #print "Stimuli started"  #[debug]
            if logEvents:  # log trial start
                eventLogger.log(eventTag, "start\t{}\t{}\t{}\t{}\t{}\t{}".format(block.thisN, present, target, num_stimuli, -1, t))  # -1 signifies unknown
        #elif stims_status == STARTED and t >= (fixationCrossDuration + 3.0):
        #    pass  # will be stopped at the end of the trial
        
        # *dummy* updates
        if t >= fixationCrossDuration and dummy.status == NOT_STARTED:
            # keep track of start time/frame for later
            dummy.tStart = t  # underestimates by a little under one frame
            dummy.frameNStart = frameN  # exact frame index
            dummy.setAutoDraw(True)
        
        # *response* updates
        if t >= fixationCrossDuration and response.status == NOT_STARTED:
            # keep track of start time/frame for later
            response.tStart = t  # underestimates by a little under one frame
            response.frameNStart = frameN  # exact frame index
            response.status = STARTED
            # keyboard checking is just starting
            response.clock.reset()  # now t=0
            event.clearEvents()
        if response.status == STARTED:
            theseKeys = event.getKeys(keyList=['y', 'n'])
            if len(theseKeys) > 0:  # at least one key was pressed
                response.keys = theseKeys[-1]  # just the last key pressed
                response.rt = response.clock.getTime()
                # was this 'correct'?
                if (response.keys == str(answer)): response.corr = 1
                else: response.corr=0
                # a response ends the routine
                continueRoutine = False
                if logEvents:  # log trial end
                    eventLogger.log(eventTag, "end\t{}\t{}\t{}\t{}\t{}\t{}".format(block.thisN, present, target, num_stimuli, response.corr, t))
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineTimer.reset()  # if we abort early the non-slip timer needs reset
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the [Esc] key)
        if event.getKeys(["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
        else:  # this Routine was not non-slip safe so reset non-slip timer
            routineTimer.reset()
    
    #-------Ending Routine "trial"-------
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    
    # stop generated stimuli
    for stim in stims:
        stopStimulus(stim)
    stims_status = NOT_STARTED
    #print "Stimuli stopped"  # [debug]
    
    # check responses
    if len(response.keys) == 0:  # No response was made
       response.keys=None
       # was no response the correct answer?!
       if str(answer).lower() == 'none': response.corr = 1  # correct non-response
       else: response.corr = 0  # failed to respond (incorrectly)
    # store data for block (TrialHandler)
    block.addData('response.keys',response.keys)
    block.addData('response.corr', response.corr)
    if response.keys != None:  # we had a response
        block.addData('response.rt', response.rt)
    thisExp.nextEntry()
    
# completed num_reps repeats of 'block'

# get names of stimulus parameters
if block.trialList in ([], [None], None):  params = []
else:  params = block.trialList[0].keys()
# save data for this loop
block.saveAsText(filename + 'block.csv', delim=',',
    stimOut=params,
    dataOut=['n','all_mean','all_std', 'all_raw'])

#------Prepare to start Routine "blank"-------
t = 0
blankClock.reset()  # clock 
frameN = -1
routineTimer.add(0.500000)
# update component parameters for each repeat
# keep track of which components have finished
blankComponents = []
blankComponents.append(blankISI)
for thisComponent in blankComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "blank"-------
continueRoutine = True
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = blankClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    # *blankISI* period
    if t >= 0.0 and blankISI.status == NOT_STARTED:
        # keep track of start time/frame for later
        blankISI.tStart = t  # underestimates by a little under one frame
        blankISI.frameNStart = frameN  # exact frame index
        blankISI.start(0.5)
    elif blankISI.status == STARTED: #one frame should pass before updating params and completing
        blankISI.complete() #finish the static period
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in blankComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the [Esc] key)
    if event.getKeys(["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

#-------Ending Routine "blank"-------
for thisComponent in blankComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)

win.close()
core.quit()
