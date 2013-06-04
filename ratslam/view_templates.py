from numpy import *

# A single view template, associated to a particular pose cell
class ViewTemplate():

  def __init__( self, pc_x, pc_y, pc_th, template ):
    self.pc_x = pc_x
    self.pc_y = pc_y
    self.pc_th = pc_th
    #self.shape = shape
    self.template = template

  def match( self, new_template ):
    # TODO: fix this, currently super terrible temporary matching method
    return sum( abs( self.template - new_template ) )

# Contains all of the individual templates
class ViewTemplates():

  def __init__( self, x_range, y_range, x_step, y_step, im_x, im_y, match_threshold ):
    self.templates = []
    self.shape = ( ( x_range[1] - x_range[0] ) / x_step, ( y_range[1] - y_range[0] ) / y_step )
    self.match_threshold = match_threshold

    # Build mask for quick subsampling
    base = arange( im_x * im_y )
    mask_t = base/im_x > y_range[0] # Remove the top section of the image
    mask_b = base/im_x < y_range[1] # Remove the bottom section of the image
    mask_l = base%im_x > x_range[0] # Remove the left section of the image
    mask_r = base%im_x < x_range[1] # Remove the right section of the image
    mask_sy = ( base / im_x - y_range[0] ) % y_step != 0 # subsample in y direction
    mask_sx = ( base % im_x - x_range[0] ) % x_step != 0 # subsample in x direction

    self.mask = mask_t & mask_b & mask_l & mask_r & mask_sy & mask_sx

  def __getitem__( self ):
    return self.templates[ index ]

  # Check if a new image matches any templates
  def match( self, input, pc_x, pc_y, pc_th ):
    template = input[self.mask].reshape(self.shape) # subsample the image
    match_val = [ t.match( template ) for t in self.templates ]

    if len( match_val ) == 0 or min( match_val ) < self.match_threshold:
      new_template = ViewTemplate( pc_x,pc_y,pc_th,template )
      self.templates.append( new_template )
      return new_template

    best_template = self.templates[ argmin( match_val ) ]

    return best_template
