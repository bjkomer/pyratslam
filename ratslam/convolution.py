import pyopencl as cl
import numpy
import math
from mako.template import Template
from scipy import ndimage

# RESULTS: Convolution using pyopencl and separated into successive 1D filters with larger buffers is the fastest

# TODO: Change the name, this does a correlation not a convolution! ( for symmetric filters it doesn't matter )
class Convolution:
  def __init__ ( self, im, fil, larger_buffer=True, sep=True, buffer_flip=False, type=numpy.float32 ):
    
    self.ctx = cl.create_some_context()
    self.queue = cl.CommandQueue( self.ctx )
    
    self.larger_buffer = larger_buffer
    self.sep = sep # whether or not the convolution is separated into 1D chunks
    self.type = type #TODO: type should just come from the input image, do a check to see if it matches the filter
    self.buffer_flip = buffer_flip # Optimization for separable convolutions where only the x direction is required
    if self.type == numpy.float32:
      self.ctype = 'float'
    elif self.type == numpy.float64:
      self.ctype = 'double'
    else:
      raise TypeError, "Data type specified is not currently supported: " + str( self.type )

    if im is not None and fil is not None:
      self.set_params( im, fil )

  def set_params( self, im, fil ):
    """Set the image and filter to be used"""

    self.filter_len = max( fil.shape )
    self.offset = int( math.floor( self.filter_len / 2 ) )
    self.dim = im.ndim
    self.fil = fil
    self.im_shape = im.shape

    self.textconf = { 'filsize' : self.filter_len,
                      'filstart' : '- ' + str( self.offset ),  # in this format to remove the compiler warning
                      'offset' : self.offset, # offset for the wrapping
                      'type' : self.ctype } # The data type in the numpy array, converted to C

    # Do the wrapping
    if self.larger_buffer:
      if self.dim == 1:
        self.buf_shape = ( self.im_shape[0] + 2 * self.offset )
        self.im = numpy.empty( self.buf_shape, dtype=self.type)
        self.im[ self.offset : -self.offset ] = im
        self.im[ : self.offset ] = im[ -2 * self.offset : -self.offset ]
        self.im[ -self.offset : ] = im[ self.offset : 2 * self.offset ]
      elif self.dim == 2:
        self.buf_shape = ( self.im_shape[0] + 2 * self.offset, self.im_shape[1] + 2 * self.offset )
        self.im = numpy.empty( self.buf_shape, dtype=self.type)
        self.im[ self.offset : -self.offset, self.offset : - self.offset ] = im
        self.im[ : self.offset, :  ] = self.im[ -2 * self.offset : -self.offset, : ]
        self.im[ -self.offset :, : ] = self.im[ self.offset : 2 * self.offset, : ]
        self.im[ :, : self.offset  ] = self.im[ :, -2 * self.offset : -self.offset ]
        self.im[ :, -self.offset : ] = self.im[ :, self.offset : 2 * self.offset ]
        self.textconf['len_y'] = self.buf_shape[1]
      elif self.dim == 3:
        self.buf_shape = ( self.im_shape[0] + 2 * self.offset, 
                           self.im_shape[1] + 2 * self.offset, 
                           self.im_shape[2] + 2 * self.offset )
        self.im = numpy.empty( self.buf_shape, dtype=self.type)
        self.im[ self.offset : -self.offset, self.offset : - self.offset, self.offset : - self.offset ] = im
        self.im[ : self.offset, :, :  ] = self.im[ -2 * self.offset : -self.offset, :, : ]
        self.im[ -self.offset :, :, : ] = self.im[ self.offset : 2 * self.offset, :, : ]
        self.im[ :, : self.offset, :  ] = self.im[ :, -2 * self.offset : -self.offset, : ]
        self.im[ :, -self.offset :, : ] = self.im[ :, self.offset : 2 * self.offset, : ]
        self.im[ :, :, : self.offset  ] = self.im[ :, :, -2 * self.offset : -self.offset ]
        self.im[ :, :, -self.offset : ] = self.im[ :, :, self.offset : 2 * self.offset ]
        self.textconf['len_y'] = self.buf_shape[1]
        self.textconf['len_z'] = self.buf_shape[2]
    else:
      self.im = im
      self.buf_shape = self.im_shape # buf_shape will be larger if a larger buffer is used for wrapping
      if self.dim == 2:
        self.textconf['len_y'] = self.buf_shape[1]
      elif self.dim == 3:
        self.textconf['len_y'] = self.buf_shape[1]
        self.textconf['len_z'] = self.buf_shape[2]

    self.set_text()

  #NOTE: OpenCL seems to flip the x and z axis from the intuitive representation, keep this in mind when coding
  def set_text( self ):
    """Allows setup of different methods of convolution"""
    
    mf = cl.mem_flags
      
    #create OpenCL buffers
    self.im_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.im)
    self.fil_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.fil)
    self.dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.im.nbytes)
    
    if self.dim == 1:
      raise NotImplemented
    elif self.dim == 2:
      if self.larger_buffer:
        if self.sep: # 2D Wrap Buffer Separable
          self.text = """
        __kernel void conv_x(__global ${type}* im, __global ${type}* fil, __global ${type}* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          ${type} sum = 0;
          
          for ( unsigned int x = 0; x < ${filsize}; x++ ) {
            sum += im[ j + ( ${offset} + i + x ${filstart} ) * ${len_y} ] * \
                   fil[ x ];
          }
          out[ j + ( i + ${offset} ) * ${len_y} ] = sum;
        }

        __kernel void conv_y(__global ${type}* im, __global ${type}* fil, __global ${type}* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          ${type} sum = 0;
          
          for ( unsigned int y = 0; y < ${filsize}; y++ ) {
            sum += im[ ( ${offset} + j + y ${filstart} ) + i * ${len_y} ] * \
                   fil[ y ];
          }
          out[ ${offset} + j + i * ${len_y} ] = sum;
        }
            """
        else:        # 2D Wrap Buffer
          self.text = """
        __kernel void conv(__global ${type}* im, __global ${type}* fil, __global ${type}* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          ${type} sum = 0;
          
          for ( unsigned int x = 0; x < ${filsize}; x++ ) {
            for ( unsigned int y = 0; y < ${filsize}; y++ ) {
              sum += im[ ( ${offset} + j + y ${filstart} ) + ( ${offset} + i + x ${filstart} ) * ${len_y} ] * \
                     fil[ y + x * ${filsize} ];
            }
          }
          out[ ${offset} + j + ( i + ${offset} ) * ${len_y} ] = sum;
        }
          """
      else:
        if self.sep: # 2D Modulo Separable
          raise NotImplemented
        else:        # 2D Modulo
          self.text = """
        __kernel void conv(__global ${type}* im, __global ${type}* fil, __global ${type}* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          ${type} sum = 0;
          
          for ( unsigned int x = 0; x < ${filsize}; x++ ) {
            for ( unsigned int y = 0; y < ${filsize}; y++ ) {
              sum += im[ ( j + y ${filstart} ) % ${len_y} + ( ( i + x ${filstart} ) % ${len_y} ) * ${len_y} ] * \
                     fil[ y + x * ${filsize} ];
            }
          }
          out[ j + i * ${len_x} ] = sum;
        }
          """
    elif self.dim == 3:
      if self.larger_buffer:
        if self.sep: # 3D Wrap Buffer Separable
          self.text = """
        __kernel void conv_x(__global ${type}* im, __global ${type}* fil, __global ${type}* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          unsigned int k = get_global_id(2);
          ${type} sum = 0;
          
          for ( unsigned int x = 0; x < ${filsize}; x++ ) {
            sum += im[ k + j * ${len_z} + ( ${offset} + i + x ${filstart} ) * ${len_z} * ${len_y} ] * \
                   fil[ x ];
          }
          out[ k + j * ${len_z} + ( i + ${offset} ) * ${len_z} * ${len_y} ] = sum;
        }

        __kernel void conv_y(__global ${type}* im, __global ${type}* fil, __global ${type}* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          unsigned int k = get_global_id(2);
          ${type} sum = 0;
          
          for ( unsigned int y = 0; y < ${filsize}; y++ ) {
            sum += im[ k + ( ${offset} + j + y ${filstart} ) * ${len_z} + i * ${len_z} * ${len_y} ] * \
                   fil[ y ];
          }
          out[ k + ( j + ${offset} ) * ${len_z} + i * ${len_z} * ${len_y} ] = sum;
        }

        __kernel void conv_z(__global ${type}* im, __global ${type}* fil, __global ${type}* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          unsigned int k = get_global_id(2);
          ${type} sum = 0;
          
          for ( unsigned int z = 0; z < ${filsize}; z++ ) {
            sum += im[ ( ${offset} + k + z ${filstart} ) + j * ${len_z} + i * ${len_z} * ${len_y} ] * \
                   fil[ z ];
          }
          out[ ${offset} + k + j * ${len_z} + i * ${len_z} * ${len_y} ] = sum;
        }
          """
        else:        # 3D Wrap Buffer
          self.text = """
        __kernel void conv(__global ${type}* im, __global ${type}* fil, __global ${type}* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          unsigned int k = get_global_id(2);
          ${type} sum = 0;
          
          for ( unsigned int x = 0; x < ${filsize}; x++ ) {
            for ( unsigned int y = 0; y < ${filsize}; y++ ) {
              for ( unsigned int z = 0; z < ${filsize}; z++ ) {
                sum += im[ ( ${offset} + k + z ${filstart} ) + \
                           ( ${offset} + j + y ${filstart} ) * ${len_z} + \
                           ( ${offset} + i + x ${filstart} ) * ${len_z} * ${len_y} ] * \
                       fil[ z + y * ${filsize} + x * ${filsize} * ${filsize} ];
              }
            }
          }
          out[ ${offset} + k + ( j + ${offset} ) * ${len_z} + ( i + ${offset} ) * ${len_z} * ${len_y} ] = sum;
        }
          """
      else:
        if self.sep: # 3D Modulo Separable
          raise NotImplemented
        else:        # 3D Modulo
          raise NotImplemented

    self.build_program()

  def build_program( self, textconf=None ):

    if textconf is None:
      textconf = self.textconf
    if textconf is None:
      text = self.text
    else:
      text = Template( self.text, output_encoding='ascii' ).render( **textconf )

    self.program = cl.Program( self.ctx, text ).build()

  def load_data( self ):

    raise NotImplemented

  def conv_im( self, im ):
    """Does the convolution on a new image provided, using the original filter and parameters"""

    self.replace_image( im )
    return self.execute()

  def execute( self ):
    if self.sep:
      if self.dim == 1:
        self.program.conv_x( self.queue, self.im_shape, None, self.im_buf, self.fil_buf, self.dest_buf )
        out = numpy.empty_like( self.im )
        cl.enqueue_read_buffer( self.queue, self.dest_buf, out ).wait()
      elif self.dim == 2:
        shape_x = self.im_shape
        shape_y = self.im_shape
        if self.larger_buffer: # need to modify the shape of the kernels slightly to account for wrapping
          shape_x = ( self.im_shape[0], self.im_shape[1] + 2 * self.offset )
          shape_y = ( self.im_shape[0] + 2 * self.offset, self.im_shape[1] )

        self.program.conv_x( self.queue, shape_x, None, self.im_buf, self.fil_buf, self.dest_buf )
        self.program.conv_y( self.queue, shape_y, None, self.dest_buf, self.fil_buf, self.im_buf )
        out = numpy.empty_like( self.im )
        cl.enqueue_read_buffer( self.queue, self.im_buf, out ).wait()
      elif self.dim == 3:
        shape_x = self.im_shape
        shape_y = self.im_shape
        shape_z = self.im_shape
        if self.larger_buffer: # need to modify the shape of the kernels slightly to account for wrapping
          shape_x = ( self.im_shape[0], self.im_shape[1] + 2 * self.offset, self.im_shape[2] + 2 * self.offset )
          shape_y = ( self.im_shape[0] + 2 * self.offset, self.im_shape[1], self.im_shape[2] + 2 * self.offset )
          shape_z = ( self.im_shape[0] + 2 * self.offset, self.im_shape[1] + 2 * self.offset, self.im_shape[2] )

        self.program.conv_x( self.queue, shape_x, None, self.im_buf, self.fil_buf, self.dest_buf )
        self.program.conv_y( self.queue, shape_y, None, self.dest_buf, self.fil_buf, self.im_buf )
        self.program.conv_z( self.queue, shape_z, None, self.im_buf, self.fil_buf, self.dest_buf )
        out = numpy.empty_like( self.im )
        cl.enqueue_read_buffer( self.queue, self.dest_buf, out ).wait()
    else:
      self.program.conv( self.queue, self.im_shape, None, self.im_buf, self.fil_buf, self.dest_buf )
      out = numpy.empty_like( self.im )
      cl.enqueue_read_buffer( self.queue, self.dest_buf, out ).wait()
    #""" disable this block for debugging
    if self.larger_buffer:
      if self.dim == 1:
        out = out[self.offset:-self.offset]
      elif self.dim == 2:
        out = out[self.offset:-self.offset,self.offset:-self.offset]
      elif self.dim == 3:
        out = out[self.offset:-self.offset,self.offset:-self.offset,self.offset:-self.offset]
    #"""
    return out

  def replace_filter( self, fil ):
    """Replaces the currently loaded filter with a new one, must be the same size"""

    if self.fil.shape == fil.shape and self.fil.dtype == fil.dtype:
      self.fil = fil
      #TODO: this might not be needed
      mf = cl.mem_flags  
      self.fil_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.fil)
    else:
      raise TypeError, "Filter must match old filter in shape and type. Old filter is of type " + \
          str( self.fil.dtype) + " and shape " + str( self.fil.shape ) + ", while new filter is of type " + \
          str( fil.dtype ) + " and shape " + str( fil.shape ) + ". To change shape/type, use new_filter()" 
  
  def new_filter( self, fil ):
    """Replaces the currently loaded filter with a new one, and updates the buffers and opencl program accordingly"""
    raise NotImplemented
  
  # TODO: needs to support non-symmetrical images
  def replace_image( self, im ):
    """Replaces the currently loaded image with a new one, must be the same size"""

    if self.im.dtype == im.dtype:
      # Do the wrapping
      if self.larger_buffer:
        if self.dim == 1:
          if self.im.shape != self.buf_shape:
            raise TypeError
          self.im = numpy.empty( self.buf_shape, dtype=self.type)
          self.im[ self.offset : -self.offset ] = im
          self.im[ : self.offset ] = im[ -2 * self.offset : -self.offset ]
          self.im[ -self.offset : ] = im[ self.offset : 2 * self.offset ]
        elif self.dim == 2:
          if self.im.shape != self.buf_shape:
            raise TypeError
          self.im = numpy.empty( self.buf_shape, dtype=self.type)
          self.im[ self.offset : -self.offset, self.offset : - self.offset ] = im
          self.im[ : self.offset, :  ] = self.im[ -2 * self.offset : -self.offset, : ]
          self.im[ -self.offset :, : ] = self.im[ self.offset : 2 * self.offset, : ]
          self.im[ :, : self.offset  ] = self.im[ :, -2 * self.offset : -self.offset ]
          self.im[ :, -self.offset : ] = self.im[ :, self.offset : 2 * self.offset ]
        elif self.dim == 3:
          if self.im.shape != self.buf_shape:
            raise TypeError
          self.im = numpy.empty( self.buf_shape, dtype=self.type)
          self.im[ self.offset : -self.offset, self.offset : - self.offset, self.offset : - self.offset ] = im
          self.im[ : self.offset, :, :  ] = self.im[ -2 * self.offset : -self.offset, :, : ]
          self.im[ -self.offset :, :, : ] = self.im[ self.offset : 2 * self.offset, :, : ]
          self.im[ :, : self.offset, :  ] = self.im[ :, -2 * self.offset :  -self.offset, : ]
          self.im[ :, -self.offset :, : ] = self.im[ :, self.offset : 2 * self.offset, : ]
          self.im[ :, :, : self.offset  ] = self.im[ :, :, -2 * self.offset : -self.offset ]
          self.im[ :, :, -self.offset : ] = self.im[ :, :, self.offset : 2 * self.offset ]
      else:
        if self.im.shape != im.shape:
          raise TypeError
        self.im = im
      #TODO: this might not be needed
      mf = cl.mem_flags
      self.im_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.im)
    else:
      raise TypeError, "Image must match old Image in shape and type. Old image is of type " + \
          str( self.im.dtype) + " and shape " + str( self.im.shape ) + ", while new image is of type " + \
          str( im.dtype ) + " and shape " + str( im.shape ) + ". To change shape/type, use new_image()" 
  
  def new_image( self, im ):
    """Replaces the currently loaded image with a new one, and updates the buffers and opencl program accordingly"""
    raise NotImplemented
