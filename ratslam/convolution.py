import pyopencl as cl
import numpy
import math
from mako.template import Template
from scipy import ndimage


class Convolution:
  def __init__ ( self, im, fil, larger_buffer=True, sep=True, type=numpy.float32 ):
    
    self.ctx = cl.create_some_context()
    self.queue = cl.CommandQueue( self.ctx )
    
    self.larger_buffer = larger_buffer
    self.sep = sep # whether or not the convolution is separated into 1D chunks
    self.type = type #TODO: type should just come from the input image, do a check to see if it matches the filter

    if im is not None and fil is not None:
      self.set_params( im, fil )

  def set_params( self, im, fil ):
    """Set the image and filter to be used"""

    self.filter_len = max( fil.shape )
    self.offset = int( math.floor( self.filter_len / 2 ) )
    self.image_len = len( im )
    self.dim = im.ndim
    self.fil = fil
    self.im_shape = im.shape

    length = self.image_len
    
    # Do the wrapping
    if self.larger_buffer:
      length = self.image_len + 2 * self.offset
      if self.dim == 1:
        self.im = numpy.empty( ( length ), dtype=self.type)
        self.im[ self.offset : -self.offset ] = im
        self.im[ : self.offset ] = im[ -2 * self.offset : -self.offset ]
        self.im[ -self.offset : ] = im[ self.offset : 2 * self.offset ]
      elif self.dim == 2:
        self.im = numpy.empty( ( length, length ), dtype=self.type)
        self.im[ self.offset : -self.offset, self.offset : - self.offset ] = im
        self.im[ : self.offset, :  ] = self.im[ length -2 * self.offset : length - self.offset, : ]
        self.im[ -self.offset :, : ] = self.im[ self.offset : 2 * self.offset, : ]
        self.im[ :, : self.offset  ] = self.im[ :, length - 2 * self.offset : length - self.offset ]
        self.im[ :, -self.offset : ] = self.im[ :, self.offset : 2 * self.offset ]
      elif self.dim == 3:
        self.im = numpy.empty( ( length, length, length ), dtype=self.type)
        self.im[ self.offset : -self.offset, self.offset : - self.offset, self.offset : - self.offset ] = im
        self.im[ : self.offset, :, :  ] = self.im[ length -2 * self.offset : length - self.offset, :, : ]
        self.im[ -self.offset :, :, : ] = self.im[ self.offset : 2 * self.offset, :, : ]
        self.im[ :, : self.offset, :  ] = self.im[ :, length - 2 * self.offset : length - self.offset, : ]
        self.im[ :, -self.offset :, : ] = self.im[ :, self.offset : 2 * self.offset, : ]
        self.im[ :, :, : self.offset  ] = self.im[ :, :, length - 2 * self.offset : length - self.offset ]
        self.im[ :, :, -self.offset : ] = self.im[ :, :, self.offset : 2 * self.offset ]
    else:
      self.im = im

    self.textconf = { 'imsize' : length,
                      'filsize' : self.filter_len,
                      'filstart' : '- ' + str( self.offset ),  # in this format to remove the compiler warning
                      'offset' : self.offset } # offset for the wrapping

    self.set_text()

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
        __kernel void conv_y(__global float* im, __global float* fil, __global float* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          float sum = 0;
          
          for ( unsigned int y = 0; y < ${filsize}; y++ ) {
            sum += im[ i + ( ${offset} + j + y ${filstart} ) * ${imsize} ] * \
                   fil[ y ];
          }
          out[ i + ( j + ${offset} ) * ${imsize} ] = sum;
        }

        __kernel void conv_x(__global float* im, __global float* fil, __global float* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          float sum = 0;
          
          for ( unsigned int x = 0; x < ${filsize}; x++ ) {
            sum += im[ ( ${offset} + i + x ${filstart} ) + j * ${imsize} ] * \
                   fil[ x ];
          }
          out[ ${offset} + i + j * ${imsize} ] = sum;
        }
            """
        else:        # 2D Wrap Buffer
          self.text = """
        __kernel void conv(__global float* im, __global float* fil, __global float* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          float sum = 0;
          
          for ( unsigned int x = 0; x < ${filsize}; x++ ) {
            for ( unsigned int y = 0; y < ${filsize}; y++ ) {
              sum += im[ ( ${offset} + i + x ${filstart} ) + ( ${offset} + j + y ${filstart} ) * ${imsize} ] * \
                     fil[ x + y * ${filsize} ];
            }
          }
          out[ ${offset} + i + ( j + ${offset} ) * ${imsize} ] = sum;
        }
          """
      else:
        if self.sep: # 2D Modulo Separable
          raise NotImplemented
        else:        # 2D Modulo
          self.text = """
        __kernel void conv(__global float* im, __global float* fil, __global float* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          float sum = 0;
          
          for ( unsigned int x = 0; x < ${filsize}; x++ ) {
            for ( unsigned int y = 0; y < ${filsize}; y++ ) {
              sum += im[ ( i + x ${filstart} ) % ${imsize} + ( ( j + y ${filstart} ) % ${imsize} ) * ${imsize} ] * \
                     fil[ x + y * ${filsize} ];
            }
          }
          out[ i + j * ${imsize} ] = sum;
        }
          """
    elif self.dim == 3:
      if self.larger_buffer:
        if self.sep: # 3D Wrap Buffer Separable
          self.text = """
        __kernel void conv_z(__global float* im, __global float* fil, __global float* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          unsigned int k = get_global_id(2);
          float sum = 0;
          
          for ( unsigned int z = 0; z < ${filsize}; z++ ) {
            sum += im[ i + j * ${imsize} + ( ${offset} + k + z ${filstart} ) * ${imsize} * ${imsize} ] * \
                   fil[ z ];
          }
          out[ i + j * ${imsize} + ( k + ${offset} ) * ${imsize} * ${imsize} ] = sum;
        }

        __kernel void conv_y(__global float* im, __global float* fil, __global float* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          unsigned int k = get_global_id(2);
          float sum = 0;
          
          for ( unsigned int y = 0; y < ${filsize}; y++ ) {
            sum += im[ i + ( ${offset} + j + y ${filstart} ) * ${imsize} + k * ${imsize} * ${imsize} ] * \
                   fil[ y ];
          }
          out[ i + ( j + ${offset} ) * ${imsize} + k * ${imsize} * ${imsize} ] = sum;
        }

        __kernel void conv_x(__global float* im, __global float* fil, __global float* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          unsigned int k = get_global_id(2);
          float sum = 0;
          
          for ( unsigned int x = 0; x < ${filsize}; x++ ) {
            sum += im[ ( ${offset} + i + x ${filstart} ) + j * ${imsize} + k * ${imsize} * ${imsize} ] * \
                   fil[ x ];
          }
          out[ ${offset} + i + j * ${imsize} + k * ${imsize} * ${imsize} ] = sum;
        }
          """
        else:        # 3D Wrap Buffer
          self.text = """
        __kernel void conv(__global float* im, __global float* fil, __global float* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          unsigned int k = get_global_id(2);
          float sum = 0;
          
          for ( unsigned int x = 0; x < ${filsize}; x++ ) {
            for ( unsigned int y = 0; y < ${filsize}; y++ ) {
              for ( unsigned int z = 0; z < ${filsize}; z++ ) {
                sum += im[ ( ${offset} + i + x ${filstart} ) + \
                           ( ${offset} + j + y ${filstart} ) * ${imsize} + \
                           ( ${offset} + k + z ${filstart} ) * ${imsize} * ${imsize} ] * \
                       fil[ x + y * ${filsize} + z * ${filsize} * ${filsize} ];
              }
            }
          }
          out[ ${offset} + i + ( j + ${offset} ) * ${imsize} + ( k + ${offset} ) * ${imsize} * ${imsize} ] = sum;
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

  def load_data( self ):

    raise NotImplemented

  def execute( self, style='2D' ):
    
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

    if self.larger_buffer:
      if self.dim == 1:
        out = out[self.offset:-self.offset]
      elif self.dim == 2:
        out = out[self.offset:-self.offset,self.offset:-self.offset]
      elif self.dim == 3:
        out = out[self.offset:-self.offset,self.offset:-self.offset,self.offset:-self.offset]

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
  
  def replace_image( self, im ):
    """Replaces the currently loaded image with a new one, must be the same size"""

    if self.im.shape == im.shape and self.im.dtype == im.dtype:
      # Do the wrapping
      if self.larger_buffer:
        length = self.image_len + 2 * self.offset
        if self.dim == 1:
          self.im = numpy.empty( ( length ), dtype=self.type)
          self.im[ self.offset : -self.offset ] = im
          self.im[ : self.offset ] = im[ -2 * self.offset : -self.offset ]
          self.im[ -self.offset : ] = im[ self.offset : 2 * self.offset ]
        elif self.dim == 2:
          self.im = numpy.empty( ( length, length ), dtype=self.type)
          self.im[ self.offset : -self.offset, self.offset : - self.offset ] = im
          self.im[ : self.offset, :  ] = self.im[ length -2 * self.offset : length - self.offset, : ]
          self.im[ -self.offset :, : ] = self.im[ self.offset : 2 * self.offset, : ]
          self.im[ :, : self.offset  ] = self.im[ :, length - 2 * self.offset : length - self.offset ]
          self.im[ :, -self.offset : ] = self.im[ :, self.offset : 2 * self.offset ]
        elif self.dim == 3:
          self.im = numpy.empty( ( length, length, length ), dtype=self.type)
          self.im[ self.offset : -self.offset, self.offset : - self.offset, self.offset : - self.offset ] = im
          self.im[ : self.offset, :, :  ] = self.im[ length -2 * self.offset : length - self.offset, :, : ]
          self.im[ -self.offset :, :, : ] = self.im[ self.offset : 2 * self.offset, :, : ]
          self.im[ :, : self.offset, :  ] = self.im[ :, length - 2 * self.offset : length - self.offset, : ]
          self.im[ :, -self.offset :, : ] = self.im[ :, self.offset : 2 * self.offset, : ]
          self.im[ :, :, : self.offset  ] = self.im[ :, :, length - 2 * self.offset : length - self.offset ]
          self.im[ :, :, -self.offset : ] = self.im[ :, :, self.offset : 2 * self.offset ]
      else:
        self.im = im
      #TODO: this might not be needed
      mf = cl.mem_flags
      self.im_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.im)
    else:
      raise TypeError, "Image must match old Image in shape and type. Old image is of type " + \
          str( self.fil.dtype) + " and shape " + str( self.fil.shape ) + ", while new image is of type " + \
          str( fil.dtype ) + " and shape " + str( fil.shape ) + ". To change shape/type, use new_image()" 
  
  def new_image( self, im ):
    """Replaces the currently loaded image with a new one, and updates the buffers and opencl program accordingly"""
    raise NotImplemented

def main():
  im_2d = numpy.random.randint(10, size=(10000,10000)).astype( numpy.float32 )
  im_3d = numpy.random.randint( 10, size=( 100, 100, 100 ) ).astype( numpy.float32 )

  fil_3d = numpy.ones( ( 10, 10, 10 ), dtype=numpy.float32 )
  fil_3d[2,2,2] = 2

  fil_2d = numpy.ones( ( 3, 3 ), dtype=numpy.float32 )
  fil_2d[1,1] = 2
  fil_2d[2,1] = 3
  fil_2d[2,2] = -1

  fil_1d = numpy.ones( ( 10 ), dtype=numpy.float32 )
  fil_1d[1] = 2

  #conv = Convolution( im=im_2d, fil=fil_2d, larger_buffer=True )
  #conv = Convolution( im=im_2d, fil=fil_1d, larger_buffer=True, sep=True )
  conv = Convolution( im=im_3d, fil=fil_1d, larger_buffer=True, sep=True )
  #conv = Convolution( im=im_3d, fil=fil_3d, larger_buffer=True, sep=False )
  conv.execute()

  # RESULT: Convolution using pyopencl and separated into successive 1D filters is the fastest

if __name__ == "__main__":
  main()
