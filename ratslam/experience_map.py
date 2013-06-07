# The experience map should only really be used for visualization,
# it is not a biologically inspired element

from numpy import *

def clip_rad_180( angle ):
  if angle > pi:
    angle -= ceil( angle / ( 2 * pi ) ) * 2 * pi
  elif angle <= -pi:
    angle += ceil( abs( angle ) / ( 2 * pi ) ) * 2 * pi
  return angle

class Experience():

  def __init__( self, pc_loc, em_loc, vt ):
    
    self.pc_x = pc_loc[0]
    self.pc_y = pc_loc[1]
    self.pc_th = pc_loc[2]

    self.vt = vt

    self.m_x = em_loc[0]
    self.m_y = em_loc[1]

  def get_point( self ):
    """Returns the (x,y) point of this experience"""

    return ( self.m_x, self.m_y )

class ExperienceMap():

  def __init__( self ):
    
    # Accumulated movement between experiences
    self.accum_delta_x = 0
    self.accum_delta_y = 0
    self.accum_delta_th = 0

    # A list of all experiences
    self.experiences = []
    self.current_exp = None

  def create( self, pc_loc, vt=None ):
    """Creates a new experience and links it to the current one"""

    exp = Experience( pc_loc, ( self.accum_delta_x, self.accum_delta_y ), vt )
    self.experiences.append( exp )
    # TODO: linking will go here
    self.current_exp = exp

  def update( self, vtrans, vrot, pc_loc, vt=None ):
    """Update the experience map with new information"""

    self.accum_delta_th = clip_rad_180( self.accum_delta_th + vrot )
    self.accum_delta_x += vtrans * cos( self.accum_delta_th )
    self.accum_delta_y += vtrans * sin( self.accum_delta_th )

    #TODO: currently leaving vt stuff out, just keeping track of location
    self.create( pc_loc )

  def get_points( self ):
    """Returns all of the (x,y) points in the experience map"""

    return [ e.get_point() for e in self.experiences ]

  def get_current_point( self ):
    """Returns the (x,y) point of the current experience"""

    return self.current_exp.get_point()
