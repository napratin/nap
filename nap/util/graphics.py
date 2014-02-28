"""Primitive and composite shapes for rendering."""

from types import ClassType
from collections import namedtuple
import xml.etree.ElementTree as ET

import numpy as np
import cv2
import cv2.cv as cv


def convert_SVG_to_image(filename=None, source=None):
  """
  Convert an SVG file or source string to OpenCV-compatible numpy image array (ARGB).
  
  Usage:
  import cv2
  import nap.util.graphics as g
  
  # Pass in filename directly
  img = g.convert_SVG_to_image("drawing.svg")
  
  # Pass in file contents
  with open("drawing.svg", "rb") as fileobj:
    img = g.convert_SVG_to_image(None, fileobj.read())  # positional args
  
  # Pass in an SVG source string
  svgSource = '<svg><rect width="300" height="100" style="fill:rgb(128,255,128);stroke-width:2;stroke:rgb(0,0,0)" /></svg>'
  img = g.convert_SVG_to_image(source=svgSource)  # keywords args
  
  cv2.imshow("Image", img)
  """
  
  if filename is None and source is None:
    print "[ERROR] convert_SVG_to_image(): Must supply either filename or SVG source"
    return None
  
  try:
    import cairo
    import rsvg
    
    # Create and SVG handler for given filename and get dimensions
    handler = rsvg.Handle(filename) if filename is not None else rsvg.Handle(None, source)  # TODO is there a better way to do this?
    #width, height, _, _ = handler.get_dimension_data()  # (width, height, float_width, float_height) ?
    width = handler.props.width
    height = handler.props.height
    
    # Create an image surface and render to it
    imageSurface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(imageSurface)
    handler.render_cairo(ctx)
    
    # Extract raw ARGB data and convert to numpy array
    image = np.ndarray((imageSurface.get_height(), imageSurface.get_width(), 4), dtype=np.uint8, buffer=imageSurface.get_data())
    return image
  except ImportError as e:
    print "[ERROR] convert_SVG_to_image(): SVG support not available (pip install cairo; apt-get install python-rsvg):", e
  
  return None


class GraphicsContext(object):
  """An encapsulation of common resources and meta information required by graphics objects."""
  
  def __init__(self):
    pass


# TODO Move this to a metaprogramming/decorators util module
def register(registry, key=None):
  """
  A simple decorator to register a type/object (obj) in a registry (key defaults to class name).
  
  Usage:
    @register(Shape.types)  # uses type name 'Point' as key
    class Point:
      ...
    
    register(Shape.types, 'circular')(Circle)  # custom key 'circular' for pre-defined type Circle
    register(Shape.types, 'circular')(Round)  # duplicate key, will print warning
  """
  def doRegister(obj):
    # Pick appropriate key if not supplied
    key_ = key  # NOTE key must not be assigned to, otherwise it will be treated as a local (Python quirk!)
    if key_ is None:
      if isinstance(obj, (type, ClassType)):  # is obj a type? (supports new-style and old-style classes)
        key_ = obj.__name__  # use type name
      else:
        key_ = obj.__class__.__name__  # it's an instance, use class name (may overwrite registry)
    
    # Check if key is in registry already (NOTE no check for whether registry is a mapping type)
    if key_ in registry:
      print "[WARNING] register(): Duplicate key: \'{}\'".format(key_)
    
    # Register object and return
    registry[key_] = obj
    return obj
  
  return doRegister


# TODO Move this to a metaprogramming module as well; serializer/deserializer instead of parser
Field = namedtuple('Field', ['required', 'default', 'parser'])  # field definition for easier parsing


# TODO Yet another generic utility function
def str2bool(v):
  return v.lower() in ("true", "yes", "t", "1")


class Shape(object):
  """Base class for all shapes with utility methods for easy inheritance and polymorphism."""
  
  # NOTE This registered type pattern and XML parsing is based on tang code.
  types = dict()
  
  @classmethod
  def fromXMLElement(cls, graphicsContext, xmlElement):
    """Create a Shape instance from XML element."""
    # NOTE Subclasses should override this to extract relevant properties from xmlElement
    #print "fromXMLElement({}, {}(), {})".format(cls.__name__, graphicsContext.__class__.__name__, xmlElement)  # [debug]
    # Delegate component creation to appropriate subclass based on XML tag, return None when invalid
    try:
      shapeType = cls if xmlElement.tag == cls.__name__ else cls.types[xmlElement.tag]
      #print "  shapeType: {}".format(shapeType.__name__)  # [debug]
      shape = shapeType(graphicsContext, **xmlElement.attrib)  # unpack dict to kwargs
      for childElement in list(xmlElement):
        shape.addChild(Shape.fromXMLElement(graphicsContext, childElement))
      return shape
    except KeyError as e:
      print "Shape.fromXMLElement(): Unregistered tag/shape type \'{}\': {}".format(xmlElement.tag, e)
    except TypeError as e:
      print "Shape.fromXMLElement(): Invalid shape type \'{}\' (not a callable?): {}".format(xmlElement.tag, e)
    # TODO Check for other exceptions
    return None
  
  __fields = None  # top-level class's fieldset (common fields), name mangled to be class-unique
  fieldSets = dict()  # collection of fieldsets from all registered types, key=type
  
  def __init__(self, graphicsContext=None, **params):
    """Initialize an instance of this (or any derived) type using a generic data-driven method."""
    # NOTE Subclasses should call Shape.__init__(self, graphicsContext, **params) first if overriding
    self.graphicsContext = graphicsContext # keep a reference to a GraphicsContext
    
    # Get fieldset from cache, or collect (only once per class)
    fieldSet = None
    if self.__class__ in self.fieldSets:
      fieldSet = self.fieldSets[self.__class__]
    else:
      fieldSetName = '_' + self.__class__.__name__ + '__fields'  # get class-specific mangled fieldset name
      #print "Class: {}, fieldSetName: {}".format(self.__class__.__name__, fieldSetName)  # [debug]
      fieldSet = None
      if hasattr(self.__class__, fieldSetName):
        # TODO Gather/loop over all such fieldSets, i.e. from ancestors as well (one-time per class?)
        fieldSet = getattr(self.__class__, fieldSetName)
      self.fieldSets[self.__class__] = fieldSet
    
    # Initialize each field with parsed or default value
    if fieldSet is not None:
      #print "Class: {}, fieldSet: {}".format(self.__class__.__name__, fieldSet)  # [debug]
      for name, field in fieldSet.iteritems():
        value = params.get(name, field.default)  # if field.default is None, we are forced to specify an explicit value
        #print "Field \'{}\': \"{}\" ({})".format(name, value, field)  # [debug]
        if field.required and value is None:
          raise ValueError("Required field {}.{} missing".format(self.__class__.__name__, name))
        setattr(self, name, value if field.parser is None else field.parser(value))  # if this crashes, we die
    
    # Initialize empty list of children
    self.children = list()
  
  def addChild(self, shape):
    """Add a shape to list of immediate children."""
    if not self == shape:  # avoid being your own child (TODO also avoid loops?)
      self.children.append(shape)
  
  def render(self, image):
    """Render this shape onto given image."""
    # NOTE Subclasses should either call-through, or call Shape.renderChildren(self, image)
    self.renderChildren(image)
  
  def renderChildren(self, image):
    """Render all child shapes."""
    for child in self.children:
      child.render(image)
    # TODO Relative positioning? When rendering children, pass in an offset? Translation, rotation, scale?
  
  def toXMLElement(self):
    """Convert this instance to an XML element."""
    # NOTE Subclasses should call Shape.toXMLElement(self) to obtain
    #   base node and then add further attributes and sub-elements
    return ET.Element(self.__class__.__name__)
  
  def toString(self, indent="", deltaIndent=" ", eol="", showChildren=False):
    """Return a brief string representation of this shape object, with optional indentation."""
    # Open
    s = indent + self.__class__.__name__ + ": {"
    deltaIndent = deltaIndent if deltaIndent is not None else ("  " if eol.endswith("\n") else " ")
    nextIndent = indent + deltaIndent
    
    # Show fields
    showFields = False
    if self.__class__ in self.fieldSets and self.fieldSets[self.__class__]:
      showFields = True
      fieldFormatStr = "{sep}{eol}{indent}{name}: {value}"
      ctr = 0  # TODO can we use enumerate()?
      for name in self.fieldSets[self.__class__].iterkeys():
        s += fieldFormatStr.format(sep=("," if ctr > 0 else ""), eol=eol, indent=nextIndent, name=name, value=getattr(self, name))
        ctr += 1
    
    # Show children
    if showChildren and self.children:
      childFormatStr = "{sep}{eol}{childStr}"
      ctr = 0  # TODO can we use enumerate()?
      for child in self.children:
        s += childFormatStr.format(sep=("," if ctr > 0 else ""), eol=eol, childStr=child.toString(indent=nextIndent, deltaIndent=deltaIndent, eol=eol, showChildren=showChildren))
        ctr += 1
    
    # Close
    if showFields or (showChildren and self.children):
      s += eol
    else:
      s += " "
    s += indent + "}"
    return s
  
  def __str__(self):
    return self.toString()
  
  def __repr__(self):
    return self.toString(deltaIndent="  ", eol="\n", showChildren=True)


@register(Shape.types)
class Point(Shape):
  __fields = dict(
    location=Field(True, "0.5 0.5", lambda valueStr: np.fromstring(valueStr, dtype=np.float32, sep=' ')),
    size=Field(True, "0.005", lambda valueStr: float(valueStr)),
    color=Field(True, "0.5 0.5 0.5", lambda valueStr: tuple(np.uint8(np.fromstring(valueStr, dtype=np.float32, sep=' ') * 255))))
  # TODO Combine and generalize some fields to Shape class
  
  def render(self, image):
    # TODO Cache some values on creation/update?
    cv2.circle(image, (int(self.location[0] * image.shape[1]), int(self.location[1] * image.shape[0])), int(self.size * image.shape[0]), (int(self.color[0]), int(self.color[1]), int(self.color[2])), cv.CV_FILLED)
    Shape.renderChildren(self, image)


@register(Shape.types)
class Line(Shape):
  __fields = dict(
    begin=Field(True, "0 0", lambda valueStr: np.fromstring(valueStr, dtype=np.float32, sep=' ')),
    end=Field(True, None, lambda valueStr: np.fromstring(valueStr, dtype=np.float32, sep=' ')),
    color=Field(True, "0.5 0.5 0.5", lambda valueStr: tuple(np.uint8(np.fromstring(valueStr, dtype=np.float32, sep=' ') * 255))),
    stroke=Field(True, "2", lambda valueStr: int(valueStr)))
  
  def render(self, image):
    cv2.line(image, (int(self.begin[0] * image.shape[1]), int(self.begin[1] * image.shape[0])), (int(self.end[0] * image.shape[1]), int(self.end[1] * image.shape[0])), (int(self.color[0]), int(self.color[1]), int(self.color[2])), self.stroke)
    Shape.renderChildren(self, image)


@register(Shape.types)
class Rectangle(Shape):
  __fields = dict(
    begin=Field(True, "0 0", lambda valueStr: np.fromstring(valueStr, dtype=np.float32, sep=' ')),
    end=Field(True, None, lambda valueStr: np.fromstring(valueStr, dtype=np.float32, sep=' ')),
    color=Field(True, "0.5 0.5 0.5", lambda valueStr: tuple(np.uint8(np.fromstring(valueStr, dtype=np.float32, sep=' ') * 255))),
    stroke=Field(True, "2", lambda valueStr: int(valueStr)),
    filled=Field(True, "true", lambda valueStr: str2bool(valueStr)))
  
  def render(self, image):
    cv2.rectangle(image, (int(self.begin[0] * image.shape[1]), int(self.begin[1] * image.shape[0])), (int(self.end[0] * image.shape[1]), int(self.end[1] * image.shape[0])), (int(self.color[0]), int(self.color[1]), int(self.color[2])), cv.CV_FILLED if self.filled else self.stroke)
    Shape.renderChildren(self, image)


@register(Shape.types)
class Circle(Shape):
  __fields = dict(
    center=Field(True, "0.5 0.5", lambda valueStr: np.fromstring(valueStr, dtype=np.float32, sep=' ')),
    radius=Field(True, "0.25", lambda valueStr: float(valueStr)),
    color=Field(True, "0.5 0.5 0.5", lambda valueStr: tuple(np.uint8(np.fromstring(valueStr, dtype=np.float32, sep=' ') * 255))),
    stroke=Field(True, "2", lambda valueStr: int(valueStr)),
    filled=Field(True, "true", lambda valueStr: str2bool(valueStr)))
  
  def render(self, image):
    cv2.circle(image, (int(self.center[0] * image.shape[1]), int(self.center[1] * image.shape[0])), int(self.radius * image.shape[0]), (int(self.color[0]), int(self.color[1]), int(self.color[2])), cv.CV_FILLED if self.filled else self.stroke)
    Shape.renderChildren(self, image)


def test_separate_shapes():
  g = GraphicsContext()
  
  pointXML = ET.Element('Point', attrib={'location': "0.4 0.8", 'color': "0.4 0.4 0.8"})
  p = Shape.fromXMLElement(g, pointXML)
  print "p: {}".format(p)
  
  circleXML = ET.Element('Circle')
  c1 = Shape.fromXMLElement(g, circleXML)
  print "c1: {}".format(c1)
  
  circleXML = ET.Element('Circle', attrib={'center': "0.75 0.75", 'radius': "0.16", 'color': "0.3 0.9 0.3"})
  c2 = Shape.fromXMLElement(g, circleXML)
  print "c2: {}".format(c2)
  
  lineStr = '<Line end="0.8 0.4" color="0.8 0.4 0.4" />'
  lineXML = ET.fromstring(lineStr)
  l = Shape.fromXMLElement(g, lineXML)
  l.addChild(c1)
  print "str(l): {}".format(str(l))
  print "repr(l):-\n{}".format(repr(l))
  
  image = np.zeros((512, 512, 3), dtype=np.uint8)
  p.render(image)
  #c1.render(image)  # c1 is a child of l, so it should get rendered there
  c2.render(image)
  l.render(image)
  cv2.imshow("Image", image)
  cv2.waitKey(3000)


shapeTreeStr = '''<Shape>
  <Circle center="0.25 0.25" radius="0.2" color="0.0 0.8 0.8" />
  <Line begin="0.25 0.6" end="0.8 0.7" color="0.8 0.4 0.4" />
  <Point location="0.75 0.25" size="0.1" color="0.7 0.7 0.0" />
</Shape>'''
def test_shape_tree():
  g = GraphicsContext()
  
  print "\ntreeStr:-\n{}".format(shapeTreeStr)
  shapeTree = Shape.fromXMLElement(g, ET.fromstring(shapeTreeStr))
  print "\nshapeTree:-\n{}".format(repr(shapeTree))
  
  image = np.zeros((512, 512, 3), dtype=np.uint8)
  shapeTree.render(image)
  cv2.imshow("Image", image)
  cv2.waitKey(3000)


shapeXMLFile = "nap/util/tests/sample-shapes.xml"
def test_file_input():
  g = GraphicsContext()
  
  print "\nFile: {}".format(shapeXMLFile)
  shapeXMLTree = ET.parse(shapeXMLFile)
  shapeXMLRoot = shapeXMLTree.getroot()
  #print "\nshapeXMLRoot: {}".format(shapeXMLRoot)
  shapeTree = Shape.fromXMLElement(g, shapeXMLRoot)
  print "\nshapeTree:-\n{}".format(repr(shapeTree))
  
  image = np.zeros((512, 512, 3), dtype=np.uint8)
  shapeTree.render(image)
  cv2.imshow("Image", image)
  cv2.waitKey(3000)


if __name__ == "__main__":
  print "Shape types registered: {}".format(Shape.types.keys())  # [debug]
  #test_separate_shapes()
  #test_shape_tree()
  test_file_input()
