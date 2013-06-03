from numpy import *
from scipy import ndimage
import glumpy

#TEMP - hardcoded for now, will come from a config file or defaults
PC_DIM_XY = 21
PC_DIM_TH = 36
PC_E_SIGMA = 1 # sigma for the excitatory gaussian kernel
PC_I_SIGMA = 2 # sigma for the inhibitory gaussian kernel
PC_E_DIM   = 7 # the size of the excitatory kernel
PC_I_DIM   = 5 # the size of the inhibitory kernel
PC_GLOBAL_INHIB = 0.00002 # value of global inhibition
PC_CELL_X_SIZE = 1
PC_C_SIZE_TH = 2 *pi / PC_DIM_TH

class PoseCellNetwork:

  def __init__( self, shape, **kwargs ):

    self.shape = shape
    self.posecells = zeros(shape)
    #NOTE: might not need if gaussian_filter works
    self.kernel_3d = self.diff_gaussian( PC_E_DIM, PC_I_DIM, PC_E_SIGMA, PC_I_SIGMA, order=3 )
    self.kernel_2d = self.diff_gaussian( PC_E_DIM, PC_I_DIM, PC_E_SIGMA, PC_I_SIGMA, order=2 )
    self.kernel_1d = self.diff_gaussian( PC_E_DIM, PC_I_DIM, PC_E_SIGMA, PC_I_SIGMA, order=1 )

    self.global_inhibition = PC_GLOBAL_INHIB
    self.pc_vtrans_scale = PC_CELL_X_SIZE
    self.pc_vrot_scale = PC_C_SIZE_TH

  # builds a 3D gaussian filter kernel with lengths of 'dim'
  def build_kernel( self, dim, sigma, order=3 ):
    if order==3:
      f = zeros( ( dim, dim, dim ) )
      center = math.floor( dim / 2 )
      for x in xrange( dim ):
        for y in xrange( dim ):
          for z in xrange( dim ):
            f[x,y,z] = 1.0 / (sigma*math.sqrt(2*pi)) * \
                math.exp( (-(x-center)**2 - (y-center)**2 - (z-center)**2 ) / (2*sigma**2))

      f /= abs(sum(f.ravel())) # normalize
      return f
    elif order==2:
      f = zeros( ( dim, dim ) )
      center = math.floor( dim / 2 )
      for x in xrange( dim ):
        for y in xrange( dim ):
          f[x,y] = 1.0 / (sigma*math.sqrt(2*pi)) * \
              math.exp( (-(x-center)**2 - (y-center)**2 ) / (2*sigma**2))

      f /= abs(sum(f.ravel())) # normalize
      return f
    elif order==1:
      f = zeros( ( dim ) )
      center = math.floor( dim / 2 )
      for x in xrange( dim ):
          f[x] = 1.0 / (sigma*math.sqrt(2*pi)) * \
              math.exp( -(x-center)**2  / (2*sigma**2))

      f /= abs(sum(f.ravel())) # normalize
      return f
    else:
      pass #TODO: put an error statement here

  # builds a 3D difference of gaussians filter kernel with lengths of 'dim'
  def diff_gaussian( self, dim_e, dim_i, sigma_e, sigma_i, order=3 ):
    dim = max(dim_e, dim_i)
    if order==3:
      f = zeros( ( dim, dim, dim ) )
      center = math.floor( dim / 2 )
      for x in xrange( dim ):
        for y in xrange( dim ):
          for z in xrange( dim ):
            f[x,y,z] = ( 1 if max(x,y,z) <= center + dim_e and min(x,y,z) >= center - dim_e else 0 ) * \
                1.0 / (sigma_e*math.sqrt(2*pi)) * \
                math.exp( (-(x-center)**2 - (y-center)**2 - (z-center)**2 ) / (2*sigma_e**2)) - \
                ( 1 if max(x,y,z) <= center + dim_i and min(x,y,z) >= center - dim_i else 0 ) * \
                1.0 / (sigma_i*math.sqrt(2*pi)) * \
                math.exp( (-(x-center)**2 - (y-center)**2 - (z-center)**2 ) / (2*sigma_i**2))

      f /= abs(sum(f.ravel())) # normalize
      return f
    elif order==2:
      f = zeros( ( dim, dim ) )
      center = math.floor( dim / 2 )
      for x in xrange( dim ):
        for y in xrange( dim ):
          f[x,y] = ( 1 if max(x,y) <= center + dim_e and min(x,y) >= center - dim_e else 0 ) * \
              1.0 / (sigma_e*math.sqrt(2*pi)) * \
              math.exp( (-(x-center)**2 - (y-center)**2 ) / (2*sigma_e**2)) - \
              ( 1 if max(x,y) <= center + dim_i and min(x,y) >= center - dim_i else 0 ) * \
              1.0 / (sigma_i*math.sqrt(2*pi)) * \
              math.exp( (-(x-center)**2 - (y-center)**2 ) / (2*sigma_i**2))

      f /= abs(sum(f.ravel())) # normalize
      return f
    elif order==1:
      f = zeros( ( dim ) )
      center = math.floor( dim / 2 )
      for x in xrange( dim ):
        f[x] = ( 1 if x <= center + dim_e and x >= center - dim_e else 0 ) * \
            1.0 / (sigma_e*math.sqrt(2*pi)) * \
            math.exp( -(x-center)**2  / (2*sigma_e**2)) - \
            ( 1 if x <= center + dim_i and x >= center - dim_i else 0 ) * \
            1.0 / (sigma_i*math.sqrt(2*pi)) * \
            math.exp( -(x-center)**2  / (2*sigma_i**2))

      f /= abs(sum(f.ravel())) # normalize
      return f
    else:
      pass #TODO: put an error statement here


  def path_integration( self, vtrans, vrot ):
    vtrans /= self.pc_vtrans_scale
    vrot /= self.pc_vrot_scale #TODO ?? is this right? 

    """
    print("after normalization")
    print(self.posecells)
    print("")
    """
    #TODO - this can be optimized better in the future
    mid = math.floor( self.shape[2] / 2 )
    for dir_pc in xrange( self.shape[2] ):
      # use a 2D gaussian filter across every theta (direction) layer, with the origin offset based on vtrans and vrot
      origin = ( vtrans*cos( (dir_pc - mid)*self.pc_vrot_scale ), vtrans*sin( (dir_pc - mid)*self.pc_vrot_scale ) )
      self.posecells[:,:,dir_pc] = \
          ndimage.correlate( self.posecells[:,:,dir_pc], self.kernel_2d, mode='wrap', origin=origin )
    # Remove any negative values
    self.posecells[self.posecells < 0] = 0
    """
    print("after xy filter")
    print(self.posecells)
    print("")
    print("")
    print(self.posecells[:,:,0])
    print(self.posecells[:,:,1])
    print(self.posecells[:,:,2])
    print(self.posecells[:,:,3])
    print(self.posecells[:,:,4])
    print("")
    """
    # and then use a 1D gaussian across the theta layers
    #TODO - this can be optimized better in the future
    origin = math.floor(vrot+.5)
    for x in xrange( self.shape[0] ):
      for y in xrange( self.shape[1] ):
        self.posecells[x,y,:] = ndimage.correlate( self.posecells[x,y,:], self.kernel_1d, mode='wrap', origin=origin )
    """
    print("after th filter")
    print(self.posecells)
    print("")
    print(self.posecells[0,0,:])
    print(self.posecells[0,1,:])
    print(self.posecells[0,2,:])
    print(self.posecells[0,3,:])
    print(self.posecells[0,4,:])
    print("")
    """
    # Remove any negative values
    self.posecells[self.posecells < 0] = 0

  # Not entirely sure what this should do yet ##???????????????????
  def get_pc_max( self ):
    (x,y,th) = unravel_index(self.posecells.argmax(), self.posecells.shape)
    return (x,y,th)
    #pass # - just copy it from ratslam-python for now

  def inject( self, energy, loc ): # loc is of the form [ x, y, z ]

    self.posecells[ loc ] += energy

  def update( self, v=(0.0, 0.0) ):

    vtrans = v[0]
    vrot = v[1]

    # 1. Internal X-Y Layer Update
    # 2. Inter-Layer Update

    #input and output the same might not work
    self.posecells = ndimage.correlate(self.posecells, self.kernel_3d, mode='wrap')

    # 3. Global Inhibition
    self.posecells[self.posecells < self.global_inhibition] = 0
    self.posecells[self.posecells >= self.global_inhibition] -= self.global_inhibition

    # 4. Normalization
    total = sum(self.posecells.ravel())
    if total != 0:
      self.posecells /= total

    # Path Integration
    self.path_integration( vtrans, vrot )

    # get the maximum pose cell
    self.max_pc = self.get_pc_max()

    #print( self.posecells ) #TODO: remove this

    return self.max_pc

  # Make the pose cell network easily printable in a human readable way
  def __str__( self ):
    #TODO: make this better
    #print ( self.posecells )
    fig = glumpy.figure( (512,512) )
    image = glumpy.Image(self.posecells[:,:,0].astype(float32))
    
    
    """
    @fig.event
    def on_draw():
      fig.clear()
      image.update()
      image.draw( x=0, y=0, z=0, width=fig.width, height=fig.height )
    """
    return "a"
    #glumpy.show() #TODO: this hangs, which is terrible, don't put it here
