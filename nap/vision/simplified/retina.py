"""Simplified retina model."""

import logging
import numpy as np
import cv2

from lumos.context import Context
from lumos.input import InputDevice, run

from ..photoreceptor import Rod, Cone
from ..retina import Projector

class Retina:
  """A multi-layered surface for hosting different types of neurons that make up a retina, simplified version."""
  
  num_rods = 10000  # humans: 90-120 million
  num_cones = 1000  # humans: 4.5-6 million
  num_bipolar_cells = 2000
  
  default_image_size = (480, 480)
  
  def __init__(self, imageSize=default_image_size, timeNow=0.0):
    # * Initialize members, parameters
    self.context = Context.getInstance()
    self.logger = logging.getLogger(__name__)
    self.logger.info("Creating simplified Retina")
    self.imageSize = imageSize
    self.timeNow = timeNow
    self.bounds = np.float32([[0.0, 0.0, 2.0], [self.imageSize[0] - 1, self.imageSize[1] - 1, 4.0]])
    self.center = (self.bounds[0] + self.bounds[1]) / 2
    self.logger.debug("Retina center: {}, image size: {}".format(self.center, self.imageSize))
    
    self.bipolarBlurSize = (5, 5)  # size of blurring kernel used when computing Bipolar cell response
    self.ganglionBlurSize = (11, 11)  # size of blurring kernel used when computing Ganglion cell response
    self.ganglionCenterKernel = np.float32(
      [ [ 1, 1, 1 ],
        [ 1, 1, 1 ],
        [ 1, 1, 1 ] ])
    self.ganglionCenterKernel /= np.sum(self.ganglionCenterKernel)  # normalize
    self.logger.info("Ganglion center kernel:\n{}".format(self.ganglionCenterKernel))  # [debug]
    self.ganglionSurroundKernel = np.float32(
      [ [ 1, 1, 1, 1, 1, 1, 1 ],
        [ 1, 1, 1, 1, 1, 1, 1 ],
        [ 1, 1, 0, 0, 0, 1, 1 ],
        [ 1, 1, 0, 0, 0, 1, 1 ],
        [ 1, 1, 0, 0, 0, 1, 1 ],
        [ 1, 1, 1, 1, 1, 1, 1 ],
        [ 1, 1, 1, 1, 1, 1, 1 ] ])
    self.ganglionSurroundKernel /= np.sum(self.ganglionSurroundKernel)  # normalize
    self.logger.info("Ganglion surround kernel:\n{}".format(self.ganglionSurroundKernel))  # [debug]
    self.ganglionCenterSurroundKernel = np.float32(
      [ [ -1, -1, -1, -1, -1, -1, -1 ],
        [ -1, -1, -1, -1, -1, -1, -1 ],
        [ -1, -1,  7,  7,  7, -1, -1 ],
        [ -1, -1,  7,  9,  7, -1, -1 ],
        [ -1, -1,  7,  7,  7, -1, -1 ],
        [ -1, -1, -1, -1, -1, -1, -1 ],
        [ -1, -1, -1, -1, -1, -1, -1 ] ])
    self.ganglionCenterSurroundKernel /= np.sum(self.ganglionCenterSurroundKernel)  # normalize
    self.logger.info("Ganglion center-surround kernel:\n{}".format(self.ganglionCenterSurroundKernel))  # [debug]
    self.ganglionKernelLevels = 4
    self.ganglionKernels = [None] * self.ganglionKernelLevels
    self.ganglionKernels[0] = self.ganglionCenterSurroundKernel
    for i in xrange(1, self.ganglionKernelLevels):
      self.ganglionKernels[i] = cv2.resize(self.ganglionKernels[i - 1], dsize=None, fx=2, fy=2)
      self.ganglionKernels[i] /= np.sum(self.ganglionKernels[i])  # normalize
    self.logger.info("Ganglion center-surround kernel sizes ({} levels): {}".format(self.ganglionKernelLevels, ", ".join("{}".format(k.shape) for k in self.ganglionKernels)))  # [debug]
    
    # * Image and related members
    # ** RGB and HSV images
    self.imageBGR = np.zeros((self.imageSize[1], self.imageSize[0], 3), dtype=np.uint8)
    self.imageHSV = np.zeros((self.imageSize[1], self.imageSize[0], 3), dtype=np.uint8)
    self.imageH = np.zeros((self.imageSize[1], self.imageSize[0], 1), dtype=np.uint8)
    self.imageS = np.zeros((self.imageSize[1], self.imageSize[0], 1), dtype=np.uint8)
    self.imageV = np.zeros((self.imageSize[1], self.imageSize[0], 1), dtype=np.uint8)
    
    # ** Freq/hue-dependent response images for rods and different cone types
    self.imageRod = np.zeros((self.imageSize[1], self.imageSize[0], 1), dtype=np.float32)
    self.imagesCone = dict()  # NOTE dict keys must match names of Cone.cone_types
    self.imagesCone['S'] = np.zeros((self.imageSize[1], self.imageSize[0], 1), dtype=np.float32)
    self.imagesCone['M'] = np.zeros((self.imageSize[1], self.imageSize[0], 1), dtype=np.float32)
    self.imagesCone['L'] = np.zeros((self.imageSize[1], self.imageSize[0], 1), dtype=np.float32)
    
    # ** Bipolar and Ganglion cell response images
    # TODO Add more Ganglion cell types with different receptive field properties (color-opponent cells)
    #   'RG' +Red    -Green
    #   'GR' +Green  -Red
    #   'RB' +Red    -Blue
    #   'BR' +Blue   -Red
    #   'BY' +Blue   -Yellow
    #   'YB' +Yellow -Blue
    #   'WK' +White  -Black (currently 'ON')
    #   'KW' +Black  -White (currently 'OFF')
    # NOTE: R = L cones, G = M cones, B = S cones
    self.imagesBipolar = dict()
    self.imagesBipolar['ON'] = np.zeros((self.imageSize[1], self.imageSize[0], 1), dtype=np.float32)
    self.imagesBipolar['OFF'] = np.zeros((self.imageSize[1], self.imageSize[0], 1), dtype=np.float32)
    self.imagesGanglion = dict()
    self.imagesGanglion['ON'] = np.zeros((self.imageSize[1], self.imageSize[0]), dtype=np.float32)
    self.imagesGanglion['OFF'] = np.zeros((self.imageSize[1], self.imageSize[0]), dtype=np.float32)
    # TODO Verify why image shapes (h, w, 1) and (h, w) are not compatible
    self.imagesGanglion['RG'] = np.zeros((self.imageSize[1], self.imageSize[0]), dtype=np.float32)
    self.imagesGanglion['GR'] = np.zeros((self.imageSize[1], self.imageSize[0]), dtype=np.float32)
    self.imagesGanglion['RB'] = np.zeros((self.imageSize[1], self.imageSize[0]), dtype=np.float32)
    self.imagesGanglion['BR'] = np.zeros((self.imageSize[1], self.imageSize[0]), dtype=np.float32)
    self.imagesGanglion['BY'] = np.zeros((self.imageSize[1], self.imageSize[0]), dtype=np.float32)
    self.imagesGanglion['YB'] = np.zeros((self.imageSize[1], self.imageSize[0]), dtype=np.float32)
    
    # ** Output image(s)
    if self.context.options.gui:
      self.imageOut = np.zeros((self.imageSize[1], self.imageSize[0], 3), dtype=np.uint8)
  
  def update(self, timeNow):
    self.timeNow = timeNow
    self.logger.debug("Retina update @ {}".format(self.timeNow))
    
    # * Get HSV
    self.imageHSV = cv2.cvtColor(self.imageBGR, cv2.COLOR_BGR2HSV)
    self.imageH, self.imageS, self.imageV = cv2.split(self.imageHSV)
    
    # * Compute Rod and Cone responses
    # TODO Need non-linear response to hue, sat, val (less dependent on sat, val for cones)
    self.imageRod = np.float32(180 - cv2.absdiff(self.imageH, Rod.rod_type.hue) % 180) * 255 * self.imageV * Rod.rod_type.responseFactor  # hack: use constant sat = 200 to make response independent of saturation
    self.imagesCone['S'] = np.float32(180 - cv2.absdiff(self.imageH, Cone.cone_types[0].hue) % 180) * self.imageS * self.imageV * Cone.cone_types[0].responseFactor
    self.imagesCone['M'] = np.float32(180 - cv2.absdiff(self.imageH, Cone.cone_types[1].hue) % 180) * self.imageS * self.imageV * Cone.cone_types[1].responseFactor
    self.imagesCone['L'] = np.float32(180 - cv2.absdiff(self.imageH, Cone.cone_types[2].hue) % 180) * self.imageS * self.imageV * Cone.cone_types[2].responseFactor
    
    # * Compute Bipolar and Ganglion cell responses
    # ** Blurring is a step that is effectively achieved in biology by horizontal cells
    imageRodBlurred = cv2.blur(self.imageRod, self.bipolarBlurSize)
    self.imagesBipolar['ON'] = np.clip(self.imageRod - 0.75 * imageRodBlurred, 0.0, 1.0)
    self.imagesBipolar['OFF'] = np.clip((1.0 - self.imageRod) - 0.75 * (1.0 - imageRodBlurred), 0.0, 1.0)  # same as (1 - ON response)?
    #imagesConeSBlurred = cv2.blur(self.imagesCone['S'], self.bipolarBlurSize)
    #imagesConeMBlurred = cv2.blur(self.imagesCone['M'], self.bipolarBlurSize)
    #imagesConeLBlurred = cv2.blur(self.imagesCone['L'], self.bipolarBlurSize)
    # ** Ganglion cells simply add up responses from a (bunch of) central bipolar cell(s) (ON/OFF) and surrounding antagonistic bipolar cells (OFF/ON)
    # *** Method 1: Center - Surround
    #imageGanglionCenterON = cv2.filter2D(self.imagesBipolar['ON'], -1, self.ganglionCenterKernel)
    #imageGanglionSurroundOFF = cv2.filter2D(self.imagesBipolar['OFF'], -1, self.ganglionSurroundKernel)
    #self.imagesGanglion['ON'] = 0.75 * imageGanglionCenterON + 0.25 * imageGanglionSurroundOFF
    # *** Method 2: Center-Surround kernel
    #self.imagesGanglion['ON'] = np.clip(cv2.filter2D(self.imagesBipolar['ON'], -1, self.ganglionCenterSurroundKernel), 0.0, 1.0)
    #self.imagesGanglion['OFF'] = np.clip(cv2.filter2D(self.imagesBipolar['OFF'], -1, self.ganglionCenterSurroundKernel), 0.0, 1.0)
    # *** Method 3: Multi-level Center-Surround kernels, taking maximum
    self.imagesGanglion['ON'].fill(0.0)
    self.imagesGanglion['OFF'].fill(0.0)
    self.imagesGanglion['RG'].fill(0.0)
    self.imagesGanglion['GR'].fill(0.0)
    self.imagesGanglion['RB'].fill(0.0)
    self.imagesGanglion['BR'].fill(0.0)
    self.imagesGanglion['BY'].fill(0.0)
    self.imagesGanglion['YB'].fill(0.0)
    
    for k in self.ganglionKernels:
      # Rod pathway
      self.imagesGanglion['ON'] = np.maximum(self.imagesGanglion['ON'], np.clip(cv2.filter2D(self.imagesBipolar['ON'], -1, k), 0.0, 1.0))
      self.imagesGanglion['OFF'] = np.maximum(self.imagesGanglion['OFF'], np.clip(cv2.filter2D(self.imagesBipolar['OFF'], -1, k), 0.0, 1.0))
      # Cone pathway
      imageRG = self.imagesCone['L'] - self.imagesCone['M']
      imageRB = self.imagesCone['L'] - self.imagesCone['S']
      imageBY = self.imagesCone['S'] - (self.imagesCone['L'] + self.imagesCone['M']) / 2
      self.imagesGanglion['RG'] = np.maximum(self.imagesGanglion['RG'], np.clip(cv2.filter2D(imageRG, -1, k), 0.0, 1.0))
      self.imagesGanglion['GR'] = np.maximum(self.imagesGanglion['GR'], np.clip(cv2.filter2D(-imageRG, -1, k), 0.0, 1.0))
      self.imagesGanglion['RB'] = np.maximum(self.imagesGanglion['RB'], np.clip(cv2.filter2D(imageRB, -1, k), 0.0, 1.0))
      self.imagesGanglion['BR'] = np.maximum(self.imagesGanglion['BR'], np.clip(cv2.filter2D(-imageRB, -1, k), 0.0, 1.0))
      self.imagesGanglion['BY'] = np.maximum(self.imagesGanglion['BY'], np.clip(cv2.filter2D(imageBY, -1, k), 0.0, 1.0))
      self.imagesGanglion['YB'] = np.maximum(self.imagesGanglion['YB'], np.clip(cv2.filter2D(-imageBY, -1, k), 0.0, 1.0))
    
    # * Compute a representative output image
    
    #self.imageOut = np.maximum(self.imagesGanglion['ON'], self.imagesGanglion['OFF'])  # max of ON- and OFF- Ganglion cells
    
    # Max of all Ganglion cell images
    self.imageOut = self.imagesGanglion['ON']
    for ganglionType, ganglionImage in self.imagesGanglion.iteritems():
      self.imageOut = np.maximum(self.imageOut, ganglionImage)
    
    #_, self.imageOut = cv2.threshold(self.imagesGanglion['ON'], 0.15, 1.0, cv2.THRESH_TOZERO)  # threshold ON- Ganglion cells only
    
    # TODO Find Ganglion cell image that contains most salient region, incorporate attention (spatial, as well as by visual feature)
    
    if self.context.options.gui:
      #cv2.imshow("Hue", self.imageH)
      #cv2.imshow("Saturation", self.imageS)
      #cv2.imshow("Value", self.imageV)
      cv2.imshow("Rod response", self.imageRod)
      cv2.imshow("S-cone response", self.imagesCone['S'])
      cv2.imshow("M-cone response", self.imagesCone['M'])
      cv2.imshow("L-cone response", self.imagesCone['L'])
      cv2.imshow("ON Bipolar cells", self.imagesBipolar['ON'])
      cv2.imshow("OFF Bipolar cells", self.imagesBipolar['OFF'])
      #cv2.imshow("ON Ganglion cells", self.imagesGanglion['ON'])
      #cv2.imshow("OFF Ganglion cells", self.imagesGanglion['OFF'])
      for ganglionType, ganglionImage in self.imagesGanglion.iteritems():
        cv2.imshow("{} Ganglion cells".format(ganglionType), ganglionImage)


class SimplifiedProjector(Projector):
  """A version of Projector that uses a simplified Retina."""
  
  def __init__(self, retina=None):
    Projector.__init__(self, retina if retina is not None else Retina())


if __name__ == "__main__":
  run(SimplifiedProjector, description="Test application that uses a SimplifiedProjector to run image input through a (simplified) Retina.")
