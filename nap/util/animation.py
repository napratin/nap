"""Utility constructs to generate animations for visual input."""

from lumos.context import Context
from lumos.input import InputDevice, run


class AnimatedInputDevice(InputDevice):
  """Reads animations from XML file, renders and sources them as an input images."""
  
  def __init__(self):
    InputDevice.__init__(self)
