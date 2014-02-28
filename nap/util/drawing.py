"""Basic drawing functions to generate a shape-based graphic."""

import types
import numpy as np
import cv2
import cv2.cv as cv

from lumos.util import KeyCode

import graphics


class Drawing(object):
  window_name = "Drawing"
  window_width, window_height = 640, 480
  window_delay = 10  # ms; determines the window update rate
  
  canvas_width, canvas_height = window_width, window_height  # TODO allow canvas size to be different from window size
  canvas_channels = 3  # 3 for color (BGR), 1 for grayscale
  canvas_dtype = np.uint8  # data type of each pixel: np.uint8, np.float32, etc.
  canvas_shape = (canvas_height, canvas_width, canvas_channels)  # numpy convention
  
  overlay_channels = canvas_channels  # number of channels in overlay, should be equal to (or less than) canvas_channels
  
  def __init__(self):
    self.graphicsContext = graphics.GraphicsContext  # common graphics context object
    self.scene = graphics.Shape(self.graphicsContext)  # root scene node
    
    self.imageCanvas = np.ones(self.canvas_shape, self.canvas_dtype) * 255  # underlying canvas image
    self.imageOverlay = np.zeros(self.canvas_shape, self.canvas_dtype)  # transient overlay
    self.imageDisplay = np.zeros(self.canvas_shape, self.canvas_dtype)  # canvas image + overlay
    self.isDisplayUpdateRequired = True
    
    self.keyMap = dict(q="Quit", d="Dump")  # a mapping for all keyboard handlers, most of them directly mapped to shape types (some special ones directly inserted here)
    self.addShapesToKeyMap()
    print "Drawing.__init__(): Key map: {{{}}}".format(", ".join(("'{}': {}".format(key, obj.__name__ if (isinstance(obj, type) or isinstance(obj, types.ClassType)) else str(obj)) for key, obj in self.keyMap.iteritems())))  # [debug]
    self.shapeType = None  # selected shape type
    self.shape = None  # current shape being edited, if any
    self.shapeParams = dict(color="0.8 0.4 0.4", stroke=2, fill=False)  # initial values for common shape parameters
    
    self.isMouseDown = False
    self.ptMouseDown = (-1, -1)
    self.ptMouseDown = (-1, -1)
  
  def run(self):
    # Open window and register callbacks
    cv2.namedWindow(self.window_name)
    cv2.setMouseCallback(self.window_name, self.onMouseEvent)
    cv2.waitKey(self.window_delay)
    
    # Main loop
    print "Drawing.run(): Starting main loop..."
    while True:
      try:
        self.updateDisplay()  # ensure we have a valid display image
        cv2.imshow(self.window_name, self.imageDisplay)
        key = cv2.waitKey(self.window_delay)
        if key != -1:
          keyCode = key & 0x00007f  # key code is in the last 8 bits, pick 7 bits for correct ASCII interpretation (8th bit indicates ?)
          keyChar = chr(keyCode) if not (key & KeyCode.SPECIAL) else None  # if keyCode is normal, convert to char (str)
          if keyCode == 0x1b or keyChar == 'q':
            break
          else:
            self.onKeyPress(key, keyChar)  # returns True if event was consumed
      except KeyboardInterrupt:
        break
    
    # Clean-up
    cv2.destroyWindow(self.window_name)
    cv2.waitKey(self.window_delay)
    print "Drawing.run(): Done."
  
  def onKeyPress(self, key, keyChar=None):
    if keyChar is not None:  # special keys may not have keyChar defined
      keyChar = keyChar.lower()  # NOTE this limits us to case-insensitive key mapping
      # Find key in map, and get mapped object
      if keyChar in self.keyMap:
        obj = self.keyMap[keyChar]  # mapped object can be anything
        # Check if mapped object is a type, and if so, is it derived from Shape?
        if (isinstance(obj, type) or isinstance(obj, types.ClassType)) and issubclass(obj, graphics.Shape):
          # TODO Cancel any active edit operation
          self.shapeType = obj  # set active type
          print "[{}]".format(self.shapeType.__name__)  # [info] common output, useful for tracking active shape changes
          return True
        elif isinstance(obj, str):  # this looks like a named command
          if obj == "Dump":
            print "[Dump]\n", repr(self.scene)
      else:
        print "[WARNING] Drawing.onKeyPress(): Unknown key: {}".format(KeyCode.describeKey(key))
    # TODO else: any special key mappings?
    return False  # unconsumed event
  
  def onMouseEvent(self, event, x, y, flags, param):
    if self.shapeType is not None:  # NOTE self.shapeType must be a Shape subclass
      xNorm = float(x) / self.canvas_width
      yNorm = float(y) / self.canvas_height
      if event == cv2.EVENT_LBUTTONDOWN:
        self.isMouseDown = True
        self.createShape(xNorm, yNorm)
        self.updateOverlay()
      elif event == cv2.EVENT_MOUSEMOVE and self.isMouseDown:  # drag (TODO check cv.CV_EVENT_FLAG_LBUTTON instead?)
        self.updateShape(xNorm, yNorm)
        self.updateOverlay()
      elif event == cv2.EVENT_LBUTTONUP:
        self.isMouseDown = False
        self.finalizeShape(xNorm, yNorm)
        self.updateCanvas()
        self.resetShape()  # do this after drawing to canvas and before clearing overlay
        self.updateOverlay()
      else:
        return False
      
      return True
    
    return False  # unconsumed event
  
  def addShapesToKeyMap(self):
    # For each shape type available, add a (key, type) mapping
    for name, shapeType in graphics.Shape.types.iteritems():
      # Find first available key
      key = None
      for letter in name:
        if letter.lower() not in self.keyMap:
          key = letter.lower()
          break
      # Add (key, type) mapping
      if key is not None:
        self.keyMap[key] = shapeType
      else:
        print "[WARNING] Drawing.addShapesToKeyMap(): No suitable key found for shape: {}".format(name)
  
  def createShape(self, x, y):
    coords = str(np.float32([x, y])).strip('[ ]')  # convert coordinate to string, as required by Shape.__init__()
    if self.shapeType == graphics.Point:
      newShapeParams = self.shapeParams.copy()  # start with a copy of common shape parameters (color, stroke, etc.)
      newShapeParams['location'] = coords  # add in any shape-specific parameters, default and optional ones can be omitted
      self.shape = self.shapeType(self.graphicsContext, **newShapeParams)  # create shape object
      return True
    elif self.shapeType == graphics.Line or self.shapeType == graphics.Rectangle:
      newShapeParams = self.shapeParams.copy()
      newShapeParams['begin'] = coords
      newShapeParams['end'] = newShapeParams['begin']
      self.shape = self.shapeType(self.graphicsContext, **newShapeParams)
      return True
    elif self.shapeType == graphics.Circle:
      newShapeParams = self.shapeParams.copy()
      newShapeParams['center'] = coords
      newShapeParams['radius'] = 0
      self.shape = self.shapeType(self.graphicsContext, **newShapeParams)
      return True
    return False
  
  def updateShape(self, x, y):
    if self.shapeType == graphics.Point:
      self.shape.location = np.float32([x, y])  # update any shape-specific parameters
      return True
    elif self.shapeType == graphics.Line or self.shapeType == graphics.Rectangle:
      self.shape.end = np.float32([x, y])
      return True
    elif self.shapeType == graphics.Circle:
      self.shape.radius = np.linalg.norm(np.float32([x, y]) - self.shape.center)
      return True
    return False
  
  def finalizeShape(self, x, y):
    # Add shape as a child of root scene node
    self.scene.addChild(self.shape)
    # TODO Push on history stack so that we can undo and stuff
    pass
  
  def resetShape(self):
    self.shape = None
  
  def updateCanvas(self):
    if self.shape is not None:
      self.shape.render(self.imageCanvas)  # active shape is drawn onto current canvas
      self.isDisplayUpdateRequired = True
  
  def updateOverlay(self):
    self.imageOverlay.fill(0)  # clear before drawing
    if self.shape is not None:
      self.shape.render(self.imageOverlay)  # only active shape is drawn
    self.isDisplayUpdateRequired = True
  
  def updateDisplay(self):
    if self.isDisplayUpdateRequired:
      self.imageDisplay[:] = self.imageCanvas[:]  # start display image with a copy of canvas image
      overlayMask = (self.imageOverlay > 0) if self.overlay_channels == 1 else np.any(self.imageOverlay > 0, axis=2)  # create mask for (non-zero) overlay regions that need to be copied (NOTE mask generated is a 1-channel 2D boolean array)
      self.imageDisplay[overlayMask] = self.imageOverlay[overlayMask]  # copy in masked pixels from overlay onto display image
      self.isDisplayUpdateRequired = False


if __name__ == "__main__":
  Drawing().run()
