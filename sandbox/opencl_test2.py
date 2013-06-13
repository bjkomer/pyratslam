import pyopencl as cl
import numpy
import math
import time
from mako.template import Template
from scipy import ndimage


class Convolution:
  def __init__ ( self, im, fil, style='2D', larger_buffer=False, sep=False ):
    
    self.ctx = cl.create_some_context()
    self.queue = cl.CommandQueue( self.ctx )

    self.filter_len = max( fil.shape )
    self.offset = int( math.floor( self.filter_len / 2 ) )
    self.image_len = len( im )
    self.dim = im.ndim
    self.larger_buffer = larger_buffer
    self.fil = fil
    self.im_shape = im.shape
    self.sep = sep # whether or not the convolution is separated into 1D chunks

    length = self.image_len
    
    # Do the wrapping
    if larger_buffer:
      length = self.image_len + 2 * self.offset
      if self.dim == 1:
        self.im = numpy.zeros( ( length ), dtype=numpy.float32)
        self.im[ self.offset : -self.offset ] = im
        self.im[ 0 : self.offset ] = im[ -2 * self.offset : -self.offset ]
        self.im[ -self.offset : ] = im[ self.offset : 2 * self.offset ]
      elif self.dim == 2:
        self.im = numpy.zeros( ( length, length ), dtype=numpy.float32)
        self.im[ self.offset : -self.offset, self.offset : - self.offset ] = im
        self.im[ : self.offset, :  ] = self.im[ length -2 * self.offset : length - self.offset, : ]
        self.im[ -self.offset :, : ] = self.im[ self.offset : 2 * self.offset, : ]
        self.im[ :, 0 : self.offset ] = self.im[ :, length - 2 * self.offset : length - self.offset ]
        self.im[ :, -self.offset : ] = self.im[ :, self.offset : 2 * self.offset ]
      elif self.dim == 3:
        #TODO: fill this in, too lazy right now
        self.im = numpy.zeros( ( length, length, length ), dtype=numpy.float32)
    else:
      self.im = im



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

    elif style == 'Separated 2D Larger Buffer':
      #create OpenCL buffers
      self.im_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.im)
      self.fil_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.fil)
      self.dest_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, self.im.nbytes)

      
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

  def build_program( self, textconf=None ):

    tstart = time.time()

    if textconf is None:
      textconf = self.textconf
    if textconf is None:
      text = self.text
    else:
      text = Template( self.text, output_encoding='ascii' ).render( **textconf )

    self.program = cl.Program( self.ctx, text ).build()

    print "Building Program...", time.time() - tstart

  def load_data( self ):

    raise NotImplemented

  def execute( self, style='2D' ):
    
    print "image\n", self.im
    print "filter\n", self.fil
    
    tstart = time.time()
    
    if self.sep:
      shape_x = self.im_shape
      shape_y = self.im_shape
      if self.larger_buffer: # need to modify the shape of the kernels slightly to account for wrapping
        shape_x = ( self.im_shape[0], self.im_shape[1] + 2 * self.offset )
        shape_y = ( self.im_shape[0] + 2 * self.offset, self.im_shape[1] )

      self.program.conv_x( self.queue, shape_x, None, self.im_buf, self.fil_buf, self.dest_buf )
      self.program.conv_y( self.queue, shape_y, None, self.dest_buf, self.fil_buf, self.im_buf )
      out = numpy.empty_like( self.im )
      cl.enqueue_read_buffer( self.queue, self.im_buf, out ).wait()
    else:
      self.program.conv( self.queue, self.im_shape, None, self.im_buf, self.fil_buf, self.dest_buf )
      out = numpy.empty_like( self.im )
      cl.enqueue_read_buffer( self.queue, self.dest_buf, out ).wait()

    
    # TEMP - put this first to show the whole buffer
    #print "image\n", self.im
    #print "filter\n", self.fil
    #print "filtered image\n", out
    
    if self.larger_buffer:
      if self.dim == 1:
        out = out[self.offset:-self.offset]
      elif self.dim == 2:
        out = out[self.offset:-self.offset,self.offset:-self.offset]
      elif self.dim == 3:
        out = out[self.offset:-self.offset,self.offset:-self.offset,self.offset:-self.offset]
    
    print "filtered image", time.time() - tstart, "\n", out

    self.out = out
    self.scipy_convolution()

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
    
    tstart = time.time()
    
    if self.sep:
      print self.fil
      #out1 = ndimage.correlate1d( input=im, weights=self.fil.tolist(), axis=0, mode='wrap', origin=0 )
      #out2 = ndimage.correlate1d( input=im, weights=self.fil.tolist(), axis=1, mode='wrap', origin=0 )
      #out = out1 * out2
      out = ndimage.correlate1d( input=im, weights=self.fil.tolist(), axis=1, mode='wrap', origin=0 )
      out = ndimage.correlate1d( input=out, weights=self.fil.tolist(), axis=0, mode='wrap', origin=0 )
    else:
      out = ndimage.correlate( im, self.fil, mode='wrap', origin=(0,0) )
    print "filtered scipy image", time.time() - tstart, "\n", out
    
    assert numpy.array_equal( out, self.out ), "The PyOpenCL result does not match with Scipy"

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
  #im = numpy.zeros( ( 10, 10 ), dtype=numpy.float32 )
  im = numpy.zeros( ( 10000, 10000 ), dtype=numpy.float32 )
  im[5,5] = 1
  im[0,0] = 2
  im[0,9] = 3
  im[1,9] = 4

  #im = numpy.arange(100).reshape((10,10))

  fil = numpy.ones( ( 3, 3 ), dtype=numpy.float32 )
  fil[1,1] = 2
  fil[2,1] = 3
  fil[2,2] = -1

  fil = numpy.ones( ( 3 ), dtype=numpy.float32 )
  fil[1] = 2

  #conv = Convolution( im=im, fil=fil, style='2D Larger Buffer', larger_buffer=True )
  conv = Convolution( im=im, fil=fil, style='Separated 2D Larger Buffer', larger_buffer=True, sep=True )
  #conv = Convolution( im=im, fil=fil, style='2D Modulo', larger_buffer=False )
  conv.build_program()
  conv.execute()

if __name__ == "__main__":
  main()
