from numpy import *
from scipy import ndimage
from scipy.special import cbrt
from convolution import Convolution
import pyopencl as cl

#TEMP - hardcoded for now, will come from a config file or defaults
PC_DIM_XY = 21
PC_DIM_TH = 36
PC_E_SIGMA = 1 # sigma for the excitatory gaussian kernel
PC_I_SIGMA = 2 # sigma for the inhibitory gaussian kernel
PC_E_DIM   = 7 #+2# the size of the excitatory kernel
PC_I_DIM   = 5 #+2# the size of the inhibitory kernel
#PC_GLOBAL_INHIB = 0.00002 # value of global inhibition
PC_GLOBAL_INHIB = 0.2 # value of global inhibition
PC_CELL_X_SIZE = .2#1
PC_C_SIZE_TH = 2.0 *pi / PC_DIM_TH

def round_up( x ):
  return ceil( x ) if x > 0 else floor( x )

class PoseCellNetwork:

  def __init__( self, shape, **kwargs ):

    self.shape = shape
    self.posecells = zeros(shape)
    #NOTE: might not need if gaussian_filter works
    self.kernel_3d = self.diff_gaussian( PC_E_DIM, PC_I_DIM, PC_E_SIGMA, PC_I_SIGMA, order=3 )
    self.kernel_2d = self.diff_gaussian( PC_E_DIM, PC_I_DIM, PC_E_SIGMA, PC_I_SIGMA, order=2 )
    self.kernel_1d = self.diff_gaussian( PC_E_DIM, PC_I_DIM, PC_E_SIGMA, PC_I_SIGMA, order=1 )
    self.kernel_1d_sep = self.diff_gaussian_separable( PC_E_DIM, PC_I_DIM, PC_E_SIGMA, PC_I_SIGMA )

    self.global_inhibition = PC_GLOBAL_INHIB
    self.pc_vtrans_scale = PC_CELL_X_SIZE
    self.pc_vrot_scale = 2.0*pi/shape[2]#PC_C_SIZE_TH
    
    #TODO: maybe change everything to float32 to make it go faster?
    #self.conv = Convolution( im=self.posecells, fil=self.kernel_1d, sep=True, type=float64 )
    #self.conv = Convolution( im=self.posecells, fil=self.kernel_1d_sep, sep=True, type=float64 )
    self.conv = Convolution( im=self.posecells, fil=self.kernel_3d, sep=False, type=float64 )
    
    filter = self.diff_gaussian_offset_2d( PC_E_SIGMA, PC_I_SIGMA, shape=(7,7), origin=(0,0) )
    #self.conv.new_filter( self.kernel_2d, dim=2 )
    self.conv.new_filter( filter, dim=2 )

  # builds a 3D gaussian filter kernel with lengths of 'dim'
  def build_kernel( self, dim, sigma, order=3 ):
    if order==3:
      f = empty( ( dim, dim, dim ) )
      center = math.floor( dim / 2 )
      for x in xrange( dim ):
        for y in xrange( dim ):
          for z in xrange( dim ):
            f[x,y,z] = 1.0 / (sigma*math.sqrt(2*pi)) * \
                math.exp( (-(x-center)**2 - (y-center)**2 - (z-center)**2 ) / (2*sigma**2))

      f /= abs(sum(f.ravel())) # normalize
      return f
    elif order==2:
      f = empty( ( dim, dim ) )
      center = math.floor( dim / 2 )
      for x in xrange( dim ):
        for y in xrange( dim ):
          f[x,y] = 1.0 / (sigma*math.sqrt(2*pi)) * \
              math.exp( (-(x-center)**2 - (y-center)**2 ) / (2*sigma**2))

      f /= abs(sum(f.ravel())) # normalize
      return f
    elif order==1:
      f = empty( ( dim ) )
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
      f = empty( ( dim, dim, dim ) )
      center = math.floor( dim / 2 )
      for x in xrange( dim ):
        for y in xrange( dim ):
          for z in xrange( dim ):
            f[x,y,z] = ( 1 if max(x,y,z) <= center + dim_e and min(x,y,z) >= center - dim_e else 0 ) * \
                1.0 / (sigma_e*math.sqrt(2*pi))**3 * \
                math.exp( (-(x-center)**2 - (y-center)**2 - (z-center)**2 ) / (2*sigma_e**2)) - \
                ( 1 if max(x,y,z) <= center + dim_i and min(x,y,z) >= center - dim_i else 0 ) * \
                1.0 / (sigma_i*math.sqrt(2*pi))**3 * \
                math.exp( (-(x-center)**2 - (y-center)**2 - (z-center)**2 ) / (2*sigma_i**2))

      f /= abs(sum(f.ravel())) # normalize
      return f
    elif order==2:
      f = empty( ( dim, dim ) )
      center = math.floor( dim / 2 )
      for x in xrange( dim ):
        for y in xrange( dim ):
          f[x,y] = ( 1 if max(x,y) <= center + dim_e and min(x,y) >= center - dim_e else 0 ) * \
              1.0 / (sigma_e * sigma_e * 2 * pi) * \
              math.exp( (-(x-center)**2 - (y-center)**2 ) / (2*sigma_e**2)) - \
              ( 1 if max(x,y) <= center + dim_i and min(x,y) >= center - dim_i else 0 ) * \
              1.0 / (sigma_i * sigma_i * 2 * pi) * \
              math.exp( (-(x-center)**2 - (y-center)**2 ) / (2*sigma_i**2))

      f /= abs(sum(f.ravel())) # normalize
      return f
    elif order==1:
      f = empty( ( dim ) )
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
  """ This one uses the Ratslam C++ gaussian formulas, which are not technically correct
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
  """
  # FIXME: this doesn't actually work at all, because DoG filters are not separable
  # builds a 1D difference of gaussians filter kernel that can be used successively to create a 3D kernel
  def diff_gaussian_separable( self, dim_e, dim_i, sigma_e, sigma_i ):
    dim = max(dim_e, dim_i)
    f = empty( ( dim ) )
    center = math.floor( dim / 2 )
    for x in xrange( dim ):
      f[x] = ( 1 if x <= center + dim_e and x >= center - dim_e else 0 ) * \
          1.0 / (sigma_e*math.sqrt(2*pi)) * \
          math.exp( -(x-center)**2  / (2*sigma_e**2)) - \
          ( 1 if x <= center + dim_i and x >= center - dim_i else 0 ) * \
          1.0 / (sigma_i*math.sqrt(2*pi)) * \
          math.exp( -(x-center)**2  / (2*sigma_i**2))

    f /= abs(sum(f.ravel())) # normalize
    f = cbrt( f ) # Take the cubed root, so the filter applied 3 times will be normalized
    return f

  def diff_gaussian_offset_2d( self, sigma_e, sigma_i, shape=( 7, 7 ), origin=( 0, 0 ) ):
    """Builds a 2D difference of gaussian kernel centered at the origin given"""
    f = empty( shape )
    x, y = meshgrid( arange( shape[0] ) - origin[0], arange( shape[1] ) - origin[1] )
    center = ( math.floor( shape[0] / 2 ), math.floor( shape[1] / 2 ) )
    f = 1.0 / ( 2* sigma_e**2 * pi ) * \
        exp( (-( x - center[0])**2 - (y - center[1])**2 ) / (2*sigma_e**2)) - \
        1.0 / ( 2 * sigma_i**2 * pi ) * \
        exp( (-( x - center[0])**2 - (y - center[1])**2 ) / (2*sigma_i**2))
    f /= abs(sum(f.ravel())) # normalize
    #f = square( cbrt( f ) ) # The result of this filter and a 1D filter should be normalized
    f = cbrt( f ) # TEMP
    return f

  def diff_gaussian_offset_1d( self, sigma_e, sigma_i, size=7, origin=0 ):
    """Builds a 1D difference of gaussian kernel centered at the origin given"""
    f = empty( size )
    x = arange( size ) - origin
    center = math.floor( size / 2 )
    f = 1.0 / (sigma_e*math.sqrt(2*pi)) * \
        exp( -square(x-center)  / (2*sigma_e**2)) - \
        1.0 / (sigma_i*math.sqrt(2*pi)) * \
        exp( -square(x-center)  / (2*sigma_i**2))
    f /= abs(sum(f.ravel())) # normalize
    f = cbrt( f ) # The result of this filter and a 2D filter should be normalized
    return f

  def filters_from_origins( self, origins, shape=( 7, 7 ) ):
    num = origins.shape[1]
    filters = empty( ( shape[0], shape[1], num ) )
    for z in xrange(num):
      filters[:,:,z] = self.diff_gaussian_offset_2d( PC_E_SIGMA, PC_I_SIGMA, shape=shape, origin=origins[:,z] )
    return filters

  def path_integration( self, vtrans, vrot ):
    vtrans /= self.pc_vtrans_scale
    vrot /= self.pc_vrot_scale #TODO ?? is this right? 

    #TODO - this can be optimized better in the future
    mid = math.floor( self.shape[2] / 2 )
    
    #"""
    dir_pc = arange( self.shape[2] ).reshape( (1, self.shape[2] ) )

    origins_exact = concatenate( ( vtrans*cos( (dir_pc - mid)*self.pc_vrot_scale ), 
                                   vtrans*sin( (dir_pc - mid)*self.pc_vrot_scale ) ), axis=0 )
    origins = concatenate( ( around( vtrans*cos( (dir_pc - mid)*self.pc_vrot_scale ) ), 
                             around( vtrans*sin( (dir_pc - mid)*self.pc_vrot_scale ) ) ), axis=0 )
    
    origins_diff = origins_exact - origins
    filters = self.filters_from_origins( origins_diff )
    
    self.posecells = self.conv.conv_im( self.posecells, axes=[0,1], radius=ceil( abs( vtrans ) ), 
                                        origins=origins, filters=filters )
    #"""
    """
    for dir_pc in xrange( self.shape[2] ):
      # use a 2D gaussian filter across every theta (direction) layer, with the origin offset based on vtrans and vrot
      origin = ( vtrans*cos( (dir_pc - mid)*self.pc_vrot_scale ), vtrans*sin( (dir_pc - mid)*self.pc_vrot_scale ) )
      #origin = ( vtrans*cos( (dir_pc)*self.pc_vrot_scale ), vtrans*sin( (dir_pc)*self.pc_vrot_scale ) )
      #origin = (4,4)
      ####print origin
      #filter = self.diff_gaussian_offset_2d( PC_E_SIGMA, PC_I_SIGMA, shape=(7,7), origin=origin )
      filter = self.diff_gaussian_offset_2d( PC_E_SIGMA, PC_I_SIGMA, shape=(12,12), origin=origin )
      #filter = self.diff_gaussian_offset_2d( PC_E_SIGMA, PC_I_SIGMA, shape=(7,7), origin=(0,0) )
      #self.conv.new_filter( filter, dim=2 )
      #self.posecells = self.conv.conv_im( self.posecells, axes=[0,1] )
      
      # Using origin shifted filter
      self.posecells[:,:,dir_pc] = \
          ndimage.correlate( self.posecells[:,:,dir_pc], filter, mode='wrap' )
      
      #self.posecells[:,:,dir_pc] = \
      #    ndimage.correlate( self.posecells[:,:,dir_pc], filter, mode='wrap', origin=origin )
      
      #self.posecells[:,:,dir_pc] = \
      #    ndimage.correlate( self.posecells[:,:,dir_pc], self.kernel_2d, mode='wrap', origin=origin )
    """
    # Remove any negative values
    self.posecells[self.posecells < 0] = 0
    
    # and then use a 1D gaussian across the theta layers
    #TODO - this can be optimized better in the future
    origin = math.floor(vrot+.5)
    #for x in xrange( self.shape[0] ):
    #  for y in xrange( self.shape[1] ):
    #    self.posecells[x,y,:] = ndimage.correlate( self.posecells[x,y,:], self.kernel_1d, mode='wrap', origin=origin )
    filter = self.diff_gaussian_offset_1d( PC_E_SIGMA, PC_I_SIGMA, size=7, origin=origin )
    self.conv.new_filter( filter, dim=1 )
    self.posecells = self.conv.conv_im( self.posecells, axes=[2] )
    #self.posecells = ndimage.correlate1d(input=self.posecells, weights=filter.tolist(), axis=2, mode='wrap')
   
    # Remove any negative values
    self.posecells[self.posecells < 0] = 0
    
  # Not entirely sure what this should do yet ##???????????????????
  def get_pc_max( self ):
    (x,y,th) = unravel_index(self.posecells.argmax(), self.posecells.shape)
    return (x,y,th)
    #pass # TODO - just copy it from ratslam-python for now

  def inject( self, energy, loc ): # loc is of the form [ x, y, z ]

    self.posecells[ loc ] += energy

  def update( self, v=(0.0, 0.0) ):

    vtrans = v[0]
    vrot = v[1]

    # 1. Internal X-Y Layer Update
    # 2. Inter-Layer Update

    #input and output the same might not work
    #self.posecells = ndimage.correlate(self.posecells, self.kernel_3d, mode='wrap')
    self.posecells = self.conv.conv_im( self.posecells )
    
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

    return self.max_pc

  # Make the pose cell network easily printable in a human readable way
  #def __str__( self ):
  #  #TODO: make this better
  #  #print ( self.posecells )
