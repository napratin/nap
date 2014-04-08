"""Ocular motion module to model eye movements."""

import logging
import numpy as np

from lumos.context import Context
from lumos.input import Projector, InputRunner


class OcularMotionSystem(object):
  """Base class for all eye movement models."""
  pass


class EmulatedOcularMotionSystem(OcularMotionSystem):
  """Eye movements emulated by a moving window over input image stream."""
  
  velocity_factor = 0.9  # per sec; weight that controls how target distance affects velocity
  max_velocity = 300  # pixels per sec; maximum velocity with which eye can move
  min_distance = 5  # pixels
  min_delta_distance = 2  # pixels
  
  def __init__(self, projector=None, timeNow=0.0):
    self.logger = logging.getLogger(self.__class__.__name__)
    self.projector = projector if projector is not None else Projector()
    self.timeNow = timeNow
    self.lastMovementTime = self.timeNow
    self.isMoving = False
    self.d = np.float32([0.0, 0.0])  # relative distance to move (x, y)
    self.v = np.float32([0.0, 0.0])  # velocity to move with (x, y)
    # TODO: Switch to (r, theta) coordinates? (makes more sense, easier to handle magnitudes)
  
  def update(self, timeNow):
    self.timeNow = timeNow
    #self.logger.debug("isMoving? %s", self.isMoving)  # [verbose]
    if self.isMoving:
      d_mag = np.linalg.norm(self.d, ord=2)  # distance magnitude
      #self.logger.debug("d: (%d, %d), mag: %d", self.d[0], self.d[1], d_mag)  # [verbose]
      if d_mag < self.min_distance:
        self.stop()
      else:
        deltaTime = self.timeNow - self.lastMovementTime
        self.v = self.velocity_factor * self.d  # target velocity, depending on distance
        v_mag = np.linalg.norm(self.v, ord=2)  # velocity magnitude
        if v_mag > self.max_velocity:
          self.v *= self.max_velocity / v_mag  # scaled down velocity to within limits
        delta = self.v * deltaTime  # delta distance to move in this update
        delta_mag = np.linalg.norm(delta, ord=2)  # delta distance magnitude
        #self.logger.debug("delta: (%d, %d), mag: %d", delta[0], delta[1], delta_mag)  # [verbose]
        if delta_mag < self.min_delta_distance:  # no point trying to move otherwise
          delta *= self.min_delta_distance / delta_mag
          delta_mag = self.min_delta_distance
        
        if delta_mag > d_mag:
          delta = self.d  # prevent overshoot
          self.d.fill(0.0)
        else:
          self.d -= delta
        
        #self.logger.debug("Moving by: %g, %g", delta[0], delta[1])  # [verbose]
        if not self.projector.shiftFocus(deltaX=int(delta[0]), deltaY=int(delta[1])):  # returns False if not movement occurred
          self.stop()
        
        self.lastMovementTime = self.timeNow  # don't update time otherwise, delta would remain too small
  
  def move(self, d):
    self.d = d
    self.logger.info("Moving to: %d, %d at %.3f", self.d[0], self.d[1], self.timeNow)  # [verbose]
    self.isMoving = True
    self.lastMovementTime = self.timeNow
  
  def stop(self):
    self.d.fill(0.0)
    self.v.fill(0.0)
    self.isMoving = False
    self.logger.debug("Stopped")  # [verbose]
  
  def reset(self):
    self.logger.debug("Reset")  # [verbose]
    offset = self.getFocusOffset()
    self.move(np.int_([-offset[0], -offset[1]]))  # move back to center
  
  def getFocusPoint(self):
    return self.projector.focusPoint
  
  def getFocusOffset(self):
    return (self.projector.focusPoint[0] - self.projector.screenSize[0] / 2, self.projector.focusPoint[1] - self.projector.screenSize[1] / 2) 
  
  def getVelocity(self):
    return self.v


# Testing
if __name__ == "__main__":
  context = Context.createInstance(description="Ocular motion testing")
  projector = Projector()
  ocular = EmulatedOcularMotionSystem(projector)
  runner = InputRunner(projector)
  
  while runner.update():
    ocular.update(context.timeNow)
    if not ocular.isMoving:
      ocular.move(np.int_([np.random.uniform(-100, 100), np.random.uniform(-100, 100)]))
  
  runner.cleanUp()
