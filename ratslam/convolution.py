import pyopencl as cl
import numpy
import math
from mako.template import Template
from scipy import ndimage

# RESULTS: Convolution using pyopencl and separated into successive 1D filters with larger buffers is the fastest

# TODO: Change the name, this does a correlation not a convolution! ( for symmetric filters it doesn't matter )
class Convolution:
  def __init__( self, im, fil, fil_1d=None, fil_2d=None, larger_buffer=True, sep=True, buffer_flip=False, type=numpy.float32 ):
    
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

    # For special convolutions, if required
    self.fil_1d = fil_1d
    self.fil_1d_origin = 0
    self.fil_2d = fil_2d
    self.fil_2d_origin = ( 0, 0 ) # offset of the center of the filter
    self.max_2d_buffer = False # just set this to false for now, it might be used in the future
    
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
    
    # Create the buffers for the specific cases where lower dimensional filters are used on high dimensional images
    # These are pretty specific to pyratslam
    if self.fil_1d is not None:
      self.fil_1d_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.fil_1d)
    if self.fil_2d is not None:
      self.fil_2d_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.fil_2d)
    
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
        
        // This does a 2D convolution across a 3D image (each plane is independent)
        // fil must be 2D and im must be 3D
        __kernel void conv_xy(__global ${type}* im, __global ${type}* fil, __global ${type}* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          unsigned int k = get_global_id(2);
          ${type} sum = 0;
          
          for ( unsigned int x = 0; x < ${filsize}; x++ ) {
            for ( unsigned int y = 0; y < ${filsize}; y++ ) {
              sum += im[ ( ${offset} + k ) + \
                         ( ${offset} + j + y ${filstart} ) * ${len_z} + \
                         ( ${offset} + i + x ${filstart} ) * ${len_z} * ${len_y} ] * \
                     fil[ y + x * ${filsize} ];
            }
          }
          out[ ${offset} + k + ( j + ${offset} ) * ${len_z} + ( i + ${offset} ) * ${len_z} * ${len_y} ] = sum;
        }
        
        // This does a 2D convolution across a 3D image (each plane is independent)
        // fil must be 2D and im must be 3D, the filter is offset by the 2D origin value
        __kernel void conv_xy_origin(__global ${type}* im, __global ${type}* fil, __global ${type}* out,
                                     __global int* origin_x, __global int* origin_y )
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          unsigned int k = get_global_id(2);
          ${type} sum = 0;
          
          for ( unsigned int x = 0; x < ${filsize}; x++ ) {
            for ( unsigned int y = 0; y < ${filsize}; y++ ) {
              sum += im[ ( ${offset} + k ) + \
                         ( origin_y[k] + ${offset} + j + y ${filstart} ) * ${len_z} + \
                         ( origin_x[k] + ${offset} + i + x ${filstart} ) * ${len_z} * ${len_y} ] * \
                     fil[ y + x * ${filsize} ];
            }
          }
          out[ ${offset} + k + ( j + ${offset} ) * ${len_z} + ( i + ${offset} ) * ${len_z} * ${len_y} ] = sum;
        }
        
        // This does a 1D convolution across a 3D image (each line is independent)
        // fil must be 1D and im must be 3D
        __kernel void conv_z(__global ${type}* im, __global ${type}* fil, __global ${type}* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          unsigned int k = get_global_id(2);
          ${type} sum = 0;
          
          for ( unsigned int z = 0; z < ${filsize}; z++ ) {
            sum += im[ ( ${offset} + k + z ${filstart} ) + \
                       ( ${offset} + j ) * ${len_z} + \
                       ( ${offset} + i ) * ${len_z} * ${len_y} ] * \
                   fil[ z ];
          }
          
          out[ ${offset} + k + ( j + ${offset} ) * ${len_z} + ( i + ${offset} ) * ${len_z} * ${len_y} ] = sum;
        }
        
        // This does a 1D convolution across a 3D image (each line is independent)
        // fil must be 1D and im must be 3D, the filter is offset by the 1D origin value
        __kernel void conv_z_origin(__global ${type}* im, __global ${type}* fil, __global ${type}* out,
                                    int origin_z )
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          unsigned int k = get_global_id(2);
          ${type} sum = 0;
          
          for ( unsigned int z = 0; z < ${filsize}; z++ ) {
            sum += im[ ( origin_z + ${offset} + k + z ${filstart} ) + \
                       ( ${offset} + j ) * ${len_z} + \
                       ( ${offset} + i ) * ${len_z} * ${len_y} ] * \
                   fil[ z ];
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

  def conv_im( self, im, axes=-1, radius=None, origins=None ):
    """Does the convolution on a new image provided, using the original filter and parameters. 
       If axis parameter is specified (in the form of a list), the convolution will only be lower 
       dimensional and only done along specific axes. This parameter is only read if sep=True"""

    if origins is None:
      self.replace_image( im )
    else:
      self.new_image( im, radius, origins )
    return self.execute( axes, origins is not None )

  def execute( self, axes=-1, origins=None ):
    
    if self.sep:
      if self.dim == 1:
        self.program.conv_x( self.queue, self.im_shape, None, self.im_buf, self.fil_buf, self.dest_buf )
        out = numpy.empty_like( self.im )
        cl.enqueue_read_buffer( self.queue, self.dest_buf, out ).wait()
      elif self.dim == 2:
        shape_x = ( self.im_shape[0], self.buf_shape[1] )
        shape_y = ( self.buf_shape[0], self.im_shape[1] )
        out = numpy.empty_like( self.im )
        if axes == -1 or ( 0 in axes and 1 in axes ):
          self.program.conv_x( self.queue, shape_x, None, self.im_buf, self.fil_buf, self.dest_buf )
          self.program.conv_y( self.queue, shape_y, None, self.dest_buf, self.fil_buf, self.im_buf )
          cl.enqueue_read_buffer( self.queue, self.im_buf, out ).wait()
        elif 0 in axes:
          self.program.conv_x( self.queue, shape_x, None, self.im_buf, self.fil_buf, self.dest_buf )
          cl.enqueue_read_buffer( self.queue, self.dest_buf, out ).wait()
        elif 1 in axes:
          self.program.conv_y( self.queue, shape_y, None, self.im_buf, self.fil_buf, self.dest_buf )
          cl.enqueue_read_buffer( self.queue, self.dest_buf, out ).wait()
        else:
          out = self.im

      elif self.dim == 3:
        shape_x = ( self.im_shape[0], self.buf_shape[1], self.buf_shape[2] )
        shape_y = ( self.buf_shape[0], self.im_shape[1], self.buf_shape[2] )
        shape_z = ( self.buf_shape[0], self.buf_shape[1], self.im_shape[2] )
        out = numpy.empty_like( self.im )
        if axes == -1 or ( 0 in axes and 1 in axes and 2 in axes ):
          self.program.conv_x( self.queue, shape_x, None, self.im_buf, self.fil_buf, self.dest_buf )
          self.program.conv_y( self.queue, shape_y, None, self.dest_buf, self.fil_buf, self.im_buf )
          self.program.conv_z( self.queue, shape_z, None, self.im_buf, self.fil_buf, self.dest_buf )
          cl.enqueue_read_buffer( self.queue, self.dest_buf, out ).wait()
        elif ( 0 in axes and 1 in axes ):
          self.program.conv_x( self.queue, shape_x, None, self.im_buf, self.fil_buf, self.dest_buf )
          self.program.conv_y( self.queue, shape_y, None, self.dest_buf, self.fil_buf, self.im_buf )
          cl.enqueue_read_buffer( self.queue, self.im_buf, out ).wait()
        elif ( 1 in axes and 2 in axes ):
          self.program.conv_y( self.queue, shape_y, None, self.im_buf, self.fil_buf, self.dest_buf )
          self.program.conv_z( self.queue, shape_z, None, self.dest_buf, self.fil_buf, self.im_buf )
          cl.enqueue_read_buffer( self.queue, self.im_buf, out ).wait()
        elif ( 0 in axes and 2 in axes ):
          self.program.conv_x( self.queue, shape_x, None, self.im_buf, self.fil_buf, self.dest_buf )
          self.program.conv_z( self.queue, shape_z, None, self.dest_buf, self.fil_buf, self.im_buf )
          cl.enqueue_read_buffer( self.queue, self.im_buf, out ).wait()
        elif 0 in axes:
          self.program.conv_x( self.queue, shape_x, None, self.im_buf, self.fil_buf, self.dest_buf )
          cl.enqueue_read_buffer( self.queue, self.dest_buf, out ).wait()
        elif 1 in axes:
          self.program.conv_y( self.queue, shape_y, None, self.im_buf, self.fil_buf, self.dest_buf )
          cl.enqueue_read_buffer( self.queue, self.dest_buf, out ).wait()
        elif 2 in axes:
          self.program.conv_z( self.queue, shape_z, None, self.im_buf, self.fil_buf, self.dest_buf )
          cl.enqueue_read_buffer( self.queue, self.dest_buf, out ).wait()
        else:
          out = self.im
    else:
      if axes == -1 or len( axes ) == self.dim:
        self.program.conv( self.queue, self.im_shape, None, self.im_buf, self.fil_buf, self.dest_buf )
        out = numpy.empty_like( self.im )
        cl.enqueue_read_buffer( self.queue, self.dest_buf, out ).wait()
      elif self.dim == 1:
        raise TypeError, "Invalid Parameters"
      elif self.dim == 2:
        raise NotImplemented, "2D lower dimensional convolution not supported yet"
      elif self.dim == 3:
        if 0 in axes and 1 in axes: # 2D convolution on each xy plane
          if origins: # List of origins for each plane is provided
            self.program.conv_xy_origin( self.queue, self.im_shape, None, 
                self.im_2d_offset_buf, self.fil_2d_buf, self.dest_buf,
                self.origins_x_buf, self.origins_y_buf )
          else:
            self.program.conv_xy( self.queue, self.im_shape, None, self.im_buf, self.fil_2d_buf, self.dest_buf )
          out = numpy.empty_like( self.im )
          cl.enqueue_read_buffer( self.queue, self.dest_buf, out ).wait()
        elif axes == [ 2 ]: # 1D convolution across each z line
          self.program.conv_z( self.queue, self.im_shape, None, self.im_buf, self.fil_1d_buf, self.dest_buf )
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

  def replace_filter( self, fil, dim=None ):
    """Replaces the currently loaded filter with a new one, must be the same size"""

    if dim is None:
      if self.fil.shape == fil.shape and self.fil.dtype == fil.dtype:
        self.fil = fil
        #TODO: this might not be needed
        mf = cl.mem_flags  
        self.fil_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.fil)
      else:
        raise TypeError, "Filter must match old filter in shape and type. Old filter is of type " + \
            str( self.fil.dtype) + " and shape " + str( self.fil.shape ) + ", while new filter is of type " + \
            str( fil.dtype ) + " and shape " + str( fil.shape ) + ". To change shape/type, use new_filter()"
    elif dim == 1 and fil.ndim == 1:
      self.fil_1d = fil
      mf = cl.mem_flags  
      self.fil_1d_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.fil_1d)
    elif dim == 2 and fil.ndim == 2:
      self.fil_2d = fil
      mf = cl.mem_flags  
      self.fil_2d_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.fil_2d)
    elif dim == 3 and fil.ndim == 3:
      raise NotImplemented
    else:
      raise TypeError, "Incorrect Parameters"
  
  def new_filter( self, fil, dim=None ):
    """Replaces the currently loaded filter with a new one, and updates the buffers and opencl program accordingly.
       If dim is specified, the filter will be saved separately for use in lower dimensional convolutions.
       For example, a 2D convolution on a 3D image"""
    if dim is None:
      if self.fil.shape == fil.shape and self.fil.dtype == fil.dtype:
        self.replace_filter( fil )
      else:
        raise NotImplemented #TODO: need to resize buffers here
    elif dim == 1 and fil.ndim == 1:
      if self.fil_1d is not None and self.fil_1d.shape == fil.shape:
        self.replace_filter( fil, dim )
      else:
        # TODO: may need sizing change stuff here later
        self.fil_1d = fil
        mf = cl.mem_flags  
        self.fil_1d_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.fil_1d)
    elif dim == 2 and fil.ndim == 2:
      if self.fil_2d is not None and self.fil_2d.shape == fil.shape:
        self.replace_filter( fil, dim )
      else:
        # TODO: may need sizing change stuff here later
        self.fil_2d = fil
        mf = cl.mem_flags  
        self.fil_2d_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.fil_2d)
    elif dim == 3 and fil.ndim == 3:
      raise NotImplemented
    else:
      raise TypeError, "Incorrect Parameters"
  
  # TODO: needs to support non-symmetrical images
  def replace_image( self, im ):
    """Replaces the currently loaded image with a new one, must be the same size"""

    if self.im.dtype == im.dtype:
      # Do the wrapping
      if self.larger_buffer:
        if self.dim == 1:
          if self.im.shape != self.buf_shape:
            raise TypeError
          #self.im = numpy.empty( self.buf_shape, dtype=self.type)
          self.im[ self.offset : -self.offset ] = im
          self.im[ : self.offset ] = im[ -2 * self.offset : -self.offset ]
          self.im[ -self.offset : ] = im[ self.offset : 2 * self.offset ]
        elif self.dim == 2:
          if self.im.shape != self.buf_shape:
            raise TypeError
          #self.im = numpy.empty( self.buf_shape, dtype=self.type)
          self.im[ self.offset : -self.offset, self.offset : - self.offset ] = im
          self.im[ : self.offset, :  ] = self.im[ -2 * self.offset : -self.offset, : ]
          self.im[ -self.offset :, : ] = self.im[ self.offset : 2 * self.offset, : ]
          self.im[ :, : self.offset  ] = self.im[ :, -2 * self.offset : -self.offset ]
          self.im[ :, -self.offset : ] = self.im[ :, self.offset : 2 * self.offset ]
        elif self.dim == 3:
          if self.im.shape != self.buf_shape:
            raise TypeError
          #self.im = numpy.empty( self.buf_shape, dtype=self.type)
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
  
  def new_image( self, im, radius=None, origins=None ):
    """Replaces the currently loaded image with a new one, and updates the buffers and opencl program accordingly
       If radius is specified then the buffers will be set to handle any origin offset of that radius"""
    if radius is None:
      raise NotImplemented
    # FIXME: this is assuming 2D radius right now
    # Do the wrapping
    if self.larger_buffer:
      if self.dim == 1:
        raise NotImplemented
      elif self.dim == 2:
        raise NotImplemented
      elif self.dim == 3:
        # Since the absolute position of the origin will move with each layer, the radius is used to for the size
        self.radius = radius #ceil( sqrt( origin[0]**2 + origin[1]**2 ) )
        if self.max_2d_buffer: # If the image remains a constant size based on maximum allowed origin
          raise NotImplemented
        else:
          # If origin is specified, the shape of the buffer will change to a form similar to this diagram:
          # (Not to scale, 'im' will likely be the largest portion by far)
          #
          #<-----buf_2d_shape[0]----->
          #____________________________
          #| ________________________  |       | offset
          #| |                       | |   |
          #| |                       | |   | radius
          #| |      ___________      | |   |
          #| |     |           |     | |       |
          #| |     |           |     | |       |
          #| |     |    im     |     | |       |  im_shape[1]
          #| |     |           |     | |       |
          #| |     |           |     | |       |
          #| |     |___________|     | |       |
          #| |                       | |   |
          #| |                       | |   | radius
          #| |_______________________| |   |
          #|___________________________|       | offset
          #
          #<>                        <>
          #offset                 offset
          #
          #  <---->             <---->
          #  radius             radius
          #
          #        <----------->
          #         im_shape[0]
          offset_f = self.offset + self.radius # Full offset
          self.buf_2d_shape = ( self.im_shape[0] + 2 * offset_f, 
                                self.im_shape[1] + 2 * offset_f, 
                                self.im_shape[2] + 2 * self.offset ) # The z axis does not need to be larger
          self.im_2d = numpy.empty( self.buf_2d_shape, dtype=self.type)

          self.im_2d[ offset_f : -offset_f , 
                      offset_f : -offset_f, 
                      self.offset : -self.offset ] = im
          
          self.im_2d[ : offset_f, :, :  ] = self.im_2d[ -2 * offset_f: -offset_f, :, : ]
          self.im_2d[ -offset_f :, :, : ] = self.im_2d[ offset_f : 2 * offset_f, :, : ]

          self.im_2d[ :, : offset_f, :  ] = self.im_2d[ :, -2 * offset_f : -offset_f, : ]
          self.im_2d[ :, -offset_f :, : ] = self.im_2d[ :, offset_f : 2 * offset_f, : ]

          self.im_2d[ :, :, : self.offset  ] = self.im_2d[ :, :, -2 * self.offset : -self.offset ]
          self.im_2d[ :, :, -self.offset : ] = self.im_2d[ :, :, self.offset : 2 * self.offset ]
          
          mf = cl.mem_flags
          self.im_2d_offset_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.im_2d)
          
          self.origins_x_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=origins[0].astype(int) )
          self.origins_y_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=origins[1].astype(int) )



