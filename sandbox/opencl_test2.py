import pyopencl as cl
import numpy
import math
from mako.template import Template
from scipy import ndimage


class Convolution:
  def __init__ ( self, im, fil, style='2D', larger_buffer=False ):
    
    self.ctx = cl.create_some_context()
    self.queue = cl.CommandQueue( self.ctx )

    self.filter_len = len( fil )
    self.offset = int( math.floor( self.filter_len / 2 ) )
    self.image_len = len( im )
    self.dim = im.ndim
    self.larger_buffer = larger_buffer
    self.fil = fil
    self.im_shape = im.shape

    if larger_buffer:
      length = self.image_len + 2 * self.offset
    else:
      length = self.image_len

    #initialize client side (CPU) arrays
    if dim == 1:
      self.im = numpy.zeros( ( length ), dtype=numpy.float32)
      self.im_shape = ( self.image_len )
    elif dim == 2:
      self.im = numpy.zeros( ( length, length ), dtype=numpy.float32)
      self.im_shape = ( self.image_len, self.image_len )
    elif dim == 3:
      self.im = numpy.zeros( ( length, length, length ), dtype=numpy.float32)
      self.im_shape = ( self.image_len, self.image_len, self.image_len )


    self.textconf = { 'imsize' : length,
                      'filsize' : self.filter_len,
                      'filstart' : '- ' + str( self.offset ),  # in this format to remove the compiler warning
                      'offset' : self.offset } # offset for the wrapping

    self.set_style( style )

  def set_style( self, style='2D' ):
    """Allows setup of different methods of convolution"""
    
    mf = cl.mem_flags
    
    if style == '2D Modulo':
      #create OpenCL buffers
      self.im_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.im)
      self.fil_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.fil)
      self.dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.im.nbytes)


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
    elif style == '2D Larger Buffer':
      #create OpenCL buffers
      self.im_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.im)
      self.fil_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.fil)
      self.dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.im.nbytes)

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

    elif style == 'Separated 2D':
      self.text = """
        __kernel void conv(__global float* im, __global float* fil, __global float* out)
        {
          unsigned int i = get_global_id(0);
          unsigned int j = get_global_id(1);
          float sum = 0;
          
          for ( int x = 0; x < ${filsize}; x++ ) {
            for ( int y = 0; y < ${filsize}; y++ ) {
              sum += im[ ( i + x ${filstart} ) % ${imsize} + ( ( j + y ${filstart} ) % ${imsize} ) * ${imsize} ] * \
                     fil[ x + y * ${filsize} ];
            }
          }
          out[ i + j * ${imsize} ] = sum;
        }
        """

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

  def execute( self, style='2D' ):
    
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
    print "image\n", self.im
    print "filter\n", self.fil
    print "filtered image\n", out

  def scipy_convolution( self ):
    """Performs the convolution without OpenCL, as a comparison"""

    im = self.im
    if self.larger_buffer:
      if self.dim == 1:
        im = self.im[self.offset:-self.offset]
      elif self.dim == 2:
        im = self.im[self.offset:-self.offset,self.offset:-self.offset]
      elif self.dim == 3:
        im = self.im[self.offset:-self.offset,self.offset:-self.offset,self.offset:-self.offset]
    
    out = ndimage.correlate( im, self.fil, mode='wrap', origin=(0,0) )
    print "filtered scipy image\n", out
    

class Example:
  def __init__ ( self ):
    
    self.ctx = cl.create_some_context()
    self.queue = cl.CommandQueue( self.ctx )

    mf = cl.mem_flags

    #initialize client side (CPU) arrays
    self.a = numpy.array(range(10), dtype=numpy.float32)
    self.b = numpy.array(range(10), dtype=numpy.float32)

    #create OpenCL buffers
    self.a_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.a)
    self.b_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.b)
    self.dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.b.nbytes)

    self.text = """
      __kernel void part1(__global float* a, __global float* b, __global float* c)
      {
        unsigned int i = get_global_id(0);
        c[i] = a[i] + 2* b[i];
      }
      """

  def build_program( self, textconf=None ):

    if textconf is None:
      text = self.text
    else:
      text = Template( self.text, output_encoding='ascii' ).render( **textconf )

    self.program = cl.Program( self.ctx, text ).build()

  def load_data( self ):

    raise NotImplemented

  def execute( self ):
    
    self.program.part1( self.queue, self.a.shape, None, self.a_buf, self.b_buf, self.dest_buf )
    c = numpy.empty_like( self.a )
    cl.enqueue_read_buffer( self.queue, self.dest_buf, c ).wait()
    print "a", self.a
    print "b", self.b
    print "c", c

def main():
  #example = Example()
  #example.build_program()
  #example.execute()

  # Set test up data
  im = numpy.zeros( ( 10, 10 ) )
  im[5,5] = 1

  fil = numpy.ones( ( 3, 3 ) )
  fil[1,1] = 2

  #conv = Convolution( style='2D Modulo' )
  conv = Convolution( im=im, fil=fil, style='2D Larger Buffer', larger_buffer=True )
  #conv = Convolution( style='2D Modulo', larger_buffer=True )
  conv.build_program()
  conv.execute()
  conv.scipy_convolution()

if __name__ == "__main__":
  main()
