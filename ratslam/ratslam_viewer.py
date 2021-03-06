import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from cv_bridge import CvBridge, CvBridgeError
import cv
from collections import deque
import rospy
from sensor_msgs.msg import Image #FIXME: test with just Image for now
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension, Int32
from rospy.numpy_msg import numpy_msg
from numpy import *

import cProfile

# NOTE: using matplotlib for now, try will glumpy later, to see if queues can be removed

#POSE_SIZE = ( 50, 50, 30 )
#POSE_SIZE = ( 25, 25, 75 )
POSE_SIZE = ( 21, 21, 36 )
# Booleans for whether or not certain displays will be shown
EM_ENABLED = True
IM_ENABLED = True #True
PC_ENABLED = True
VT_ENABLED = False
TM_ENABLED = True

class RatslamViewer( ):
  
  def __init__ ( self, root="navbot" ):
    self.root = root       # Root name of ROS topic
    self.em_data = deque() # Experience Map
    self.im_data = deque() # Captured Images
    self.pc_data = deque() # Pose Cell Network
    self.vt_data = deque() # Visual Templates
    self.tm_data = deque() # Template Matches
    
    self.bridge = CvBridge() # For converting ROS images into a readable format
    
    if PC_ENABLED:
      self.pc_fig = plt.figure( 1 )
      self.pc_ax = self.pc_fig.add_subplot(111, projection='3d')
      self.pc_ax.set_title("Pose Cell Network")
      self.pc_ax.set_xlim3d([0, POSE_SIZE[0]])
      self.pc_ax.set_ylim3d([0, POSE_SIZE[1]])
      self.pc_ax.set_zlim3d([0, POSE_SIZE[2]])
      self.pc_ax.set_autoscale_on( False )
      self.pc_fig.show()
      self.pc_fig.canvas.draw()
    
    if IM_ENABLED:
      self.im_fig = plt.figure( 2 )
      self.im_ax = self.im_fig.add_subplot(111)
      self.im_ax.set_title("Visual Field")
      self.im_im = self.im_ax.imshow( zeros( ( 256, 256, 4 ) ) ) # Blank starting image
      self.im_fig.show()
      self.im_im.axes.figure.canvas.draw()
    
    if EM_ENABLED:
      self.em_fig = plt.figure( 3 )
      self.em_ax = self.em_fig.add_subplot(111)
      self.em_ax.set_title("Experience Map")
      self.em_ax.hold( True )
      self.em_fig.show()
      self.em_fig.canvas.draw()

    if VT_ENABLED:
      self.vt_fig = plt.figure( 4 )
      self.vt_ax = self.vt_fig.add_subplot(111)
      self.vt_ax.set_title("Visual Template")
      self.vt_im = self.vt_ax.imshow( zeros( ( 256, 256 ) ), cmap=plt.cm.gray ) # Blank starting image
      self.vt_fig.show()
      self.vt_fig.canvas.draw()

    if TM_ENABLED:
      self.tm_fig = plt.figure( 5 )
      self.tm_ax = self.tm_fig.add_subplot(111)
      self.tm_ax.set_title("Template Matches")
      self.tm_ax.hold( True )
      self.tm_fig.show()
      self.tm_fig.canvas.draw()

  def em_callback( self, data ):
    self.em_data.append( ( data.x, data.y ) )

  def im_callback( self, data ):
    cv_im = self.bridge.imgmsg_to_cv( data, "rgba8" )
    im = asarray( cv_im )
    self.im_data.append( im )

  def pc_callback( self, data ):
    xlen = data.layout.dim[0].size
    ylen = data.layout.dim[1].size
    zlen = data.layout.dim[2].size
    pc_np = array( data.data, dtype=float64 ).reshape( ( xlen, ylen, zlen ) )
    self.pc_data.append( pc_np )

  def vt_callback( self, data ):
    cv_vt = self.bridge.imgmsg_to_cv( data, "mono8" )
    vt = asarray( cv_vt )
    self.im_data.append( vt )
  
  def tm_callback( self, data ):
    self.tm_data.append( data.data ) #FIXME: find the right data type for this and make sure format is correct

  def run( self ):
    rospy.init_node('ratslam_viewer', anonymous=True)
    if EM_ENABLED:
      sub_em = rospy.Subscriber( self.root + '/experiencemap', Pose2D, self.em_callback)
    if IM_ENABLED:
      sub_im = rospy.Subscriber( self.root + '/camera/image', Image, self.im_callback)
    if PC_ENABLED:
      sub_pc = rospy.Subscriber( self.root + '/posecells', numpy_msg(Float64MultiArray), self.pc_callback)
    if VT_ENABLED:
      sub_vt = rospy.Subscriber( self.root + '/visualtemplate', Image, self.vt_callback)
    if TM_ENABLED:
      sub_tm = rospy.Subscriber( self.root + '/templatematches', Int32, self.tm_callback)
    em_prev_xy = ( 0, 0 )

    pc_scatter = None
    tm_count = 0 # The index of the current template match
    while not rospy.is_shutdown():
      if self.em_data:
        em = self.em_data.popleft()
        # TODO: Incorporate direction being faced into the display
        self.em_ax.plot( [ em_prev_xy[0], em[0] ], [ em_prev_xy[1], em[1] ],'b')
        em_prev_xy = em
        self.em_fig.canvas.draw()
      if self.im_data:
        im = self.im_data.popleft()
        self.im_im.set_data( im )
        self.im_im.axes.figure.canvas.draw()
      if self.pc_data:
        pc = self.pc_data.popleft()
        # TODO: do conversion from sparse matrix here
        #pc = pc_sparse
        pc_index = nonzero( pc>.002 )
        pc_value = pc[ pc_index ] * 100
        if pc_scatter is not None:
          pc_scatter.remove()
        pc_scatter = self.pc_ax.scatter(pc_index[0],pc_index[1],pc_index[2],s=pc_value)
        
        self.pc_fig.canvas.draw()
      if self.vt_data:
        vt = self.vt_data.popleft()
        self.vt_im.set_data( vt )
        self.vt_im.axes.figure.canvas.draw()
      if self.tm_data:
        tm = self.tm_data.popleft()
        self.tm_ax.scatter( tm_count, tm, c='r')
        self.tm_fig.canvas.draw()
        tm_count += 1

def main():
  viewer = RatslamViewer( root="navbot" )
  viewer.run()

if __name__ == "__main__":
  cProfile.run('main()','pstats_viewer.out') # TEMP: write out profiling stats
  #main()
