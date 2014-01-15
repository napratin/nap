"""Quadtree implementation with associated helpers."""

class Rect(object):
  def __init__(self, left, top, right, bottom):
    self.left = left
    self.top = top
    self.right = right
    self.bottom = bottom
  
  def contains(self, location):
    return self.left <= location[0] <= self.right and \
           self.top  <= location[1] <= self.bottom
  
  def __str__(self):
    return "Rect: (({r.left}, {r.top}), ({r.right}, {r.bottom}))".format(r=self)


class QuadTree(object):
  """An implementation of a point-item quad-tree.
  
  Based on: http://www.pygame.org/wiki/QuadTree
  
  Items being stored are assumed to be point objects with bounding box the 
  same as the point location. Each item is inserted into a single quadrant.
  """
  def __init__(self, items, depth=8, bounding_rect=None):
    """Creates a quad-tree.
  
    @param items:
      A sequence of items to store in the quad-tree. Note that these
      items must possess a 2-D location attribute.
      
    @param depth:
      The maximum recursion depth.
      
    @param bounding_rect:
      The bounding rect (left, top, right, bottom) of all of the items
      in the quad-tree. Specifying this speeds up the init process.
    """
    # The sub-quadrants are empty to start with.
    self.nw = self.ne = self.se = self.sw = None
    
    # If we've reached the maximum depth then insert all items into this
    # quadrant.
    depth -= 1
    if depth == 0:
      self.items = items
      return
 
    # Find this quadrant's centre.
    if bounding_rect:
      l, t, r, b = bounding_rect
    else:
      # If there isn't a bounding rect, then calculate it from the items.
      l = min(item.location[0] for item in items)
      t = min(item.location[1] for item in items)
      r = max(item.location[0] for item in items)
      b = max(item.location[1] for item in items)
    cx = self.cx = (l + r) * 0.5
    cy = self.cy = (t + b) * 0.5
    
    self.items = []
    nw_items = []
    ne_items = []
    se_items = []
    sw_items = []
    
    for item in items:
      # Point items must lie in only one of the 4 quadrants
      if item.location[0] < cx:
        if item.location[1] < cy:
          nw_items.append(item)
        else:
          sw_items.append(item)
      elif item.location[0] >= cx:
        if item.location[1] < cy:
          ne_items.append(item)
        else:
          se_items.append(item)
    
    # Create the sub-quadrants, recursively.
    if nw_items:
      self.nw = QuadTree(nw_items, depth, (l, t, cx, cy))
    if ne_items:
      self.ne = QuadTree(ne_items, depth, (cx, t, r, cy))
    if se_items:
      self.se = QuadTree(se_items, depth, (cx, cy, r, b))
    if sw_items:
      self.sw = QuadTree(sw_items, depth, (l, cy, cx, b))
  
  def hit(self, rect):
    """Returns the items that overlap a bounding rectangle.
    
    Returns the set of all items in the quad-tree that overlap with a
    bounding rectangle.
    
    @param rect:
      The bounding rectangle being tested against the quad-tree. This
      must possess a contains method that checks a 2-item tuple/list.
    """
    
    # Find the hits at the current level
    hits = set(item for item in self.items if rect.contains(item.location))
    
    # Recursively check the lower quadrants
    if self.nw and rect.left < self.cx and rect.top < self.cy:
      hits |= self.nw.hit(rect)
    if self.sw and rect.left < self.cx and rect.bottom >= self.cy:
      hits |= self.sw.hit(rect)
    if self.ne and rect.right >= self.cx and rect.top < self.cy:
      hits |= self.ne.hit(rect)
    if self.se and rect.right >= self.cx and rect.bottom >= self.cy:
      hits |= self.se.hit(rect)
    
    return hits
