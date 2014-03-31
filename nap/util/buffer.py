"""Buffers for communication with high-level (cognitive) architectures, and possibly between modules.

A buffer can generally contain a single object. Bidirectional buffers are
special in that they keep two objects, one for *input* and one for *output*.
The interpretation of input and output depend who *owns* the buffer.
E.g. an input buffer is set by external entities and read by the owner,
whereas an output buffer is set by the owner and read externally.

"""

class BufferAccessError(RuntimeError):
  """Indicates an invalid buffer access attempt."""
  pass


class Buffer(object):
  """A simple buffer."""
  
  def __init__(self, value=None):
    self.value = value  # don't use set() as it may be denied() in some types
  
  def get(self, clear=False):
    if clear:
      temp = self.value
      self.clear()
      return temp
    else:
      return self.value
  
  def set(self, value):
    self.value = value
  
  def clear(self):
    self.value = None
  
  def denied(self, *args, **kwargs):
    raise BufferAccessError("Operation denied")
  
  def __str__(self):
    return str(self.value)


class InputBuffer(Buffer):
  """A buffer type that specializes its interface for input."""
  
  get_in = Buffer.get  # internal
  get = Buffer.denied  # external


class OutputBuffer(Buffer):
  """A buffer type that specializes its interface for output."""
  
  set_out = Buffer.set  # internal
  set = Buffer.denied  # external


class BidirectionalBuffer(object):
  """A special kind of buffer that contains two simple buffers for simultaneous bidirectional data transfer."""
  
  def __init__(self, value_in=None, value_out=None):
    self.buffer_in = InputBuffer(value_in)
    self.buffer_out = OutputBuffer(value_out)
    
    # Internal interface
    self.get_in = self.buffer_in.get_in
    self.set_out = self.buffer_out.set_out
    self.get_out = self.buffer_out.get  # for convenience, to read back set value in output buffer
    self.clear = self.buffer_in.clear  # NOTE internally, we cannot set an input buffer, but we can clear it
    
    # External interface: get/set methods have a slightly different interpretation now
    self.get = self.buffer_out.get
    self.set = self.buffer_in.set
  
  def __str__(self):
    return "in: {}, out: {}".format(str(self.buffer_in.value), str(self.buffer_out.value))


# Testing
if __name__ == "__main__":
  # NOTE Here we are testing the external interfaces only, i.e. get, set methods
  i = InputBuffer('foo')
  o = OutputBuffer('bar')
  print "i: {}, o: {}".format(i, o)
  
  i.set('moo')  # ok, can set input buffers
  print "o.get(clear=True):", o.get(clear=True)  # ok, can read output buffers, and optionally clear them
  
  #print "i.get:", i.get()  # error, cannot read input buffers
  #o.set('moo')  # error, cannot set output buffers
  
  print "i: {}, o: {}".format(i, o)
  
  b = BidirectionalBuffer(5, 6)
  print "b:", b
  print "b.get():", b.get()  # ok
  b.set(9)  # ok
  print "b:", b
