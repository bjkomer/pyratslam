# Reads ROS Pose messages and uses these as input to the pose network
from posecell_network import PoseCellNetwork
from view_templates import ViewTemplates
from experience_map import ExperienceMap
from visual_odometer import SimpleVisualOdometer
from numpy import *
from scipy import ndimage, sparse
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pylab
import rospy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension, Int32
from rospy.numpy_msg import numpy_msg
#from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image #FIXME: test with just Image for now
from cv_bridge import CvBridge, CvBridgeError
import cv
from collections import deque

import cProfile
import time
import rosbag
import datetime # For naming rosbag files

# TEMP: testing with smaller things to make it run faster
#POSE_SIZE = ( 50, 50, 30 )
#POSE_SIZE = ( 25, 25, 75 )
POSE_SIZE = ( 21, 21, 36 )
IM_SIZE = ( 256, 256 ) #( 512, 512 )
#IM_SIZE = ( 128, 128 ) #( 512, 512 )
#X_RANGE = ( 64, 192 ) #( 128, 384 )
#Y_RANGE = ( 64, 192 ) #( 128, 384 )
X_RANGE = ( 32, 96 )
Y_RANGE = ( 32, 96 )
X_STEP = 2
Y_STEP = 2
MATCH_THRESHOLD = 45000 #larger value=easier match#10 # the smaller the value, the easier it is to make a match
ODOM_FREQ = 10 # Frequency in which odomentry messages are published
USE_VISUAL_ODOMETRY = False #True
RECORD_ROSBAG = False # Record the output to a rosbag file

class RatslamRos():

  def __init__( self ):
    
    self.im_count = 0

    # Initialize the Pose Cell Network
    self.pcn = PoseCellNetwork(shape=POSE_SIZE) 
    self.pc_count = 0
    self.twist_data = deque()
    self.odom_freq = ODOM_FREQ
    midpoint = (math.floor(POSE_SIZE[0]/2),math.floor(POSE_SIZE[1]/2),math.floor(POSE_SIZE[2]/2))
    self.pcn.inject( 1, midpoint )

    # Set up layout for ROS message
    dim = [ MultiArrayDimension( label="x", size=POSE_SIZE[0], stride=POSE_SIZE[0] * POSE_SIZE[1] * POSE_SIZE[2] ),
            MultiArrayDimension( label="y", size=POSE_SIZE[1], stride=POSE_SIZE[1] * POSE_SIZE[2] ),
            MultiArrayDimension( label="th", size=POSE_SIZE[2], stride=POSE_SIZE[2] ) ]
    self.pc_layout = MultiArrayLayout(dim=dim)

    # Initialize the View Templates
    # TODO: put reasonable values here
    self.vts = ViewTemplates( x_range=X_RANGE, y_range=Y_RANGE, 
                              x_step=X_STEP, y_step=Y_STEP, 
                              im_x=IM_SIZE[0], im_y=IM_SIZE[1], 
                              match_threshold=MATCH_THRESHOLD)
    self.vt_count = 0
    self.bridge = CvBridge()

    # Initialize the Experience Map
    self.em = ExperienceMap()
    self.em_count = 0
    
    self.vis_odom_data = deque()

    rospy.init_node( 'posecells', anonymous=True )
    if not USE_VISUAL_ODOMETRY:
      self.sub_odom = rospy.Subscriber( 'navbot/odom',Odometry,self.odom_callback )
    self.sub_vis = rospy.Subscriber( 'navbot/camera/image',Image,self.vis_callback )
    self.pub_pc = rospy.Publisher( 'navbot/posecells', numpy_msg(Float64MultiArray) )
    self.pub_em = rospy.Publisher( 'navbot/experiencemap', Pose2D )
    # Set up publisher for the template matches (sends just the index value of a match)
    self.pub_tm = rospy.Publisher( 'navbot/templatematches', Int32 )

    # Set up a visual odometer
    self.vis_odom = SimpleVisualOdometer()

    if RECORD_ROSBAG:
      date = datetime.datetime.now()
      self.bag = rosbag.Bag( '../testdata/Output-{0}-{1}-{2}-{3}.bag'.format( date.month, date.day, 
                                                                              date.hour, date.minute ),'w' )

  # This is called whenever new visual information is received
  def vis_callback( self, data ):

    cv_im = self.bridge.imgmsg_to_cv( data, "mono8" )
    im = asarray( cv_im )

    pc_max = self.pcn.get_pc_max()
    template_match = self.vts.match( input=im,pc_x=pc_max[0],pc_y=pc_max[1],pc_th=pc_max[2] )
    index = template_match.get_index()
    #TEMP - just inject with this template for now
    #self.pcn.inject( .02, template_match.location() )
    #self.pcn.inject( .2, template_match.location() )
  
    # Send the template match index to the viewer
    index_msg = Int32()
    index_msg.data = index
    self.pub_tm.publish( index_msg )

    if RECORD_ROSBAG:
      self.bag.write( 'navbot/camera/image', data )
      self.bag.write( 'navbot/templatematches', index_msg )

    # If using visual odometry, update the odometry information using this new image
    if USE_VISUAL_ODOMETRY:
      delta = self.vis_odom.update( im )
      self.vis_odom_data.append( delta )

  # This is called whenever new odometry information is received
  def odom_callback( self, data ):
    twist = data.twist.twist
    # If there is barely any motion, don't bother flooding the queue with it
    if abs(twist.linear.x) > 0.001 or abs(twist.angular.z) > 0.001:
      self.twist_data.append(twist)

    if RECORD_ROSBAG:
      self.bag.write( 'navbot/odom', data )

  def update_posecells( self, vtrans, vrot ):
    self.pcn.update( ( vtrans, vrot ) )
    pc_max = self.pcn.get_pc_max()
    self.em.update( vtrans, vrot, pc_max )

    #pc = sparse.csr_matrix( self.pcn.posecells ) # Create scipy sparse matrix
    pc = self.pcn.posecells

    em_msg = Pose2D()
    em_msg.x, em_msg.y = self.em.get_current_point() # TODO: add theta as well
    
    self.pub_pc.publish( self.pc_layout, pc.ravel() )
    self.pub_em.publish( em_msg )

    if RECORD_ROSBAG:
      self.bag.write( 'navbot/experiencemap', em_msg )
      self.bag.write( 'navbot/posecells', numpy_msg(self.pc_layout, pc.ravel()) )

  def run( self ):

    count = 0
    start_time = time.time()
    while not rospy.is_shutdown():
      if self.twist_data:
        twist = self.twist_data.popleft()
        vtrans = twist.linear.x / self.odom_freq
        vrot = twist.angular.z / self.odom_freq
        self.update_posecells( vtrans, vrot )
        count += 1
      if self.vis_odom_data:
        vtrans, vrot = self.vis_odom_data.popleft()
        self.update_posecells( vtrans, vrot )
        count += 1
      #elif count > 350: #FIXME: temporary break for speed testing
      #  print( "Total Time:" )
      #  print( time.time() - start_time )
      #  break

def main():
  ratslam = RatslamRos()
  ratslam.run()

if __name__ == "__main__":
  cProfile.run('main()','pstats.out') # TEMP: write out profiling stats
  #main()
