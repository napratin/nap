"""Ocular motion module to model eye movements."""

import numpy as np

from lumos.context import Context
from lumos.input import Projector, InputRunner


class OcularMotion(object):
  """Base class for all eye movement models."""
  pass


class EmulatedOcularMotion(OcularMotion):
  """Eye movements emulated by a moving window over input image stream."""
  
  velocity_factor = 0.9  # per sec; weight that controls how target distance affects velocity
  max_velocity = 100  # pixels per sec; maximum velocity with which eye can move
  min_distance = 20  # pixels
  
  def __init__(self, projector=None, timeNow=0.0):
    self.projector = projector if projector is not None else Projector()
    self.timeNow = timeNow
    self.lastUpdated = self.timeNow
    self.isMoving = False
    self.d = np.float32([0.0, 0.0])  # relative distance to move (x, y)
    self.v = np.float32([0.0, 0.0])  # velocity to move with (x, y)
  
  def update(self, timeNow):
    self.timeNow = timeNow
    if self.isMoving:
      d_mag = np.linalg.norm(self.d)  # distance magnitude
      if d_mag < self.min_distance:
        self.stop()
      else:
        deltaTime = self.timeNow - self.lastUpdated
        self.v = self.velocity_factor * self.d  # target velocity, depending on distance
        v_mag = np.linalg.norm(self.v)  # velocity magnitude
        if v_mag > self.max_velocity:
          self.v *= self.max_velocity / v_mag  # scaled down velocity to within limits
        delta = self.v * deltaTime  # delta distance to move in this update
        delta_mag = np.linalg.norm(delta)  # delta distance magnitude
        if delta_mag > d_mag:
          delta = self.d  # prevent overshoot
        # TODO: Add some noise to delta to make it more realistic
        self.d -= delta
        
        lastfocusPoint = projector.focusPoint  # store last focus point to check if there was any movement
        self.projector.shiftFocus(deltaX=delta[0], deltaY=delta[1])  # TODO: have shiftFocus return an indication of movement
        if self.projector.focusPoint == lastfocusPoint:
          self.d.fill(0.0)
          self.v.fill(0.0)
          self.stop()
    self.lastUpdated = self.timeNow
  
  def move(self, d):
    self.d = d
    self.isMoving = True
  
  def stop(self):
    self.isMoving = False


# Testing
if __name__ == "__main__":
  context = Context.createInstance(description="Ocular motion testing")
  projector = Projector()
  ocular = EmulatedOcularMotion(projector)
  runner = InputRunner(projector)
  
  while runner.update():
    ocular.update(context.timeNow)
    if not ocular.isMoving:
      ocular.move(np.float32([np.random.uniform(-100, 100), np.random.uniform(-100, 100)]))
  
  runner.cleanUp()
