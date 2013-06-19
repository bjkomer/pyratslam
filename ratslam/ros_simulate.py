# Reads ROS Pose messages and uses these as input to the pose network
from posecell_network import PoseCellNetwork
from view_templates import ViewTemplates
from experience_map import ExperienceMap
from numpy import *
from scipy import ndimage, sparse
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pylab
import rospy
from geometry_msgs.msg import Twist, TwistWithCovariance, Pose2D
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from rospy.numpy_msg import numpy_msg
#from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image #FIXME: test with just Image for now
from cv_bridge import CvBridge, CvBridgeError
import cv
from collections import deque

import cProfile

POSE_SIZE = ( 50, 50, 30 )
#POSE_SIZE = ( 50, 50, 50 ) # FIXME: temp change to make it symmetric
#POSE_SIZE = ( 5, 5, 5 ) # FIXME: temp change to make it symmetric
IM_SIZE = ( 256, 256 ) #( 512, 512 )
X_RANGE = ( 64, 192 ) #( 128, 384 )
Y_RANGE = ( 64, 192 ) #( 128, 384 )
X_STEP = 2
Y_STEP = 2
MATCH_THRESHOLD = 10 # the smaller the value, the easier it is to make a match
ODOM_FREQ = 10 # Frequency in which odomentry messages are published

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

    pc = self.pcn.posecells
    pc_index = nonzero(pc>.002)
    pc_value = pc[pc_index] * 100
  
  # This is called whenever new visual information is received
  def vis_callback( self, data ):

    cv_im = self.bridge.imgmsg_to_cv( data, "mono8" )
    im = asarray( cv_im )

    pc_max = self.pcn.get_pc_max()
    template_match = self.vts.match( input=im,pc_x=pc_max[0],pc_y=pc_max[1],pc_th=pc_max[2] )
    #TEMP - just inject with this template for now
    #self.pcn.inject( .02, template_match.location() )
  
  # This is called whenever new odometry information is received
  def odom_callback( self, data ):
    twist = data.twist.twist
    # If there is barely any motion, don't bother flooding the queue with it
    if abs(twist.linear.x) > 0.001 or abs(twist.angular.z) > 0.001:
      self.twist_data.append(twist)

  def run( self ):

    rospy.init_node( 'posecells', anonymous=True )
    sub_odom = rospy.Subscriber( 'navbot/odom',Odometry,self.odom_callback )
    sub_vis = rospy.Subscriber( 'navbot/camera/image',Image,self.vis_callback )
    pub_pc = rospy.Publisher( 'navbot/posecells', numpy_msg(Float64MultiArray) )
    pub_em = rospy.Publisher( 'navbot/experiencemap', Pose2D )

    while not rospy.is_shutdown():
      if len(self.twist_data) > 0:
        twist = self.twist_data.popleft()
        vtrans = twist.linear.x / self.odom_freq
        vrot = twist.angular.z / self.odom_freq
        self.pcn.update( ( vtrans, vrot ) )
        pc_max = self.pcn.get_pc_max()
        self.em.update( vtrans, vrot, pc_max )

        #pc = sparse.csr_matrix( self.pcn.posecells ) # Create scipy sparse matrix
        pc = self.pcn.posecells

        em_msg = Pose2D()
        em_msg.x, em_msg.y = self.em.get_current_point() # TODO: add theta as well
        
        pub_pc.publish( self.pc_layout, pc.ravel() )
        pub_em.publish( em_msg )

def main():
  ratslam = RatslamRos()
  ratslam.run()

if __name__ == "__main__":
  cProfile.run('main()','pstats.out') # TEMP: write out profiling stats
  #main()
