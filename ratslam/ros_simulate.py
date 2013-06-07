# Reads ROS Pose messages and uses these as input to the pose network
from posecell_network import PoseCellNetwork
from view_templates import ViewTemplates
from experience_map import ExperienceMap
from numpy import *
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pylab
import rospy
from geometry_msgs.msg import Twist, TwistWithCovariance
from nav_msgs.msg import Odometry
#from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image #FIXME: test with just Image for now
from cv_bridge import CvBridge, CvBridgeError
import cv
from collections import deque

POSE_SIZE = ( 50, 50, 30 )
IM_SIZE = ( 256, 256 ) #( 512, 512 )
X_RANGE = ( 64, 192 ) #( 128, 384 )
Y_RANGE = ( 64, 192 ) #( 128, 384 )
X_STEP = 2
Y_STEP = 2
MATCH_THRESHOLD = 10 # the smaller the value, the easier it is to make a match
DISP_IMAGE = True # whether or not visual images will be displayed (they are quite slow)
VT_REFRESH_PERIOD = 15 # how fast the image displayed refreshes, lower number means faster
EM_REFRESH_PERIOD = 0 # how fast the experience map refreshes, lower number means faster
PC_REFRESH_PERIOD = 0 # how fast the experience map refreshes, lower number means faster
ODOM_FREQ = 10 # Frequency in which odomentry messages are published
NUM_IMAGES = 531 # The number of images to read before the program will exit, this is for cProfile purposes only


class RatslamRos():

  def __init__( self ):
    
    self.im_count = 0

    # Initialize the Pose Cell Network
    self.pcn = PoseCellNetwork(shape=POSE_SIZE) 
    self.pc_count = 0
    self.twist_data = deque()
    self.odom_freq = ODOM_FREQ
    self.pc_fig = plt.figure(1)
    self.pc_ax = self.pc_fig.add_subplot(111, projection='3d')
    midpoint = (math.floor(POSE_SIZE[0]/2),math.floor(POSE_SIZE[1]/2),math.floor(POSE_SIZE[2]/2))
    self.pcn.inject( 1, midpoint )
    self.pc_ax.set_xlim3d([0, POSE_SIZE[0]])
    self.pc_ax.set_ylim3d([0, POSE_SIZE[1]])
    self.pc_ax.set_zlim3d([0, POSE_SIZE[2]])
    self.pc_ax.hold(False)

    # Initialize the View Templates
    # TODO: put reasonable values here
    self.vts = ViewTemplates( x_range=X_RANGE, y_range=Y_RANGE, 
                              x_step=X_STEP, y_step=Y_STEP, 
                              im_x=IM_SIZE[0], im_y=IM_SIZE[1], 
                              match_threshold=MATCH_THRESHOLD)
    self.image_data = deque() #TODO: these should really not be stored anywhere for long
    self.vt_count = 0
    self.vts_fig = plt.figure(2)
    self.vts_ax = self.vts_fig.add_subplot(111)
    self.bridge = CvBridge()

    # Initialize the Experience Map
    self.em = ExperienceMap()
    self.em_count = 0
    self.em_fig = plt.figure(3)
    self.em_ax = self.em_fig.add_subplot(111)
    self.em_ax.hold(True)

    plt.ion()
    plt.show()


    pc = self.pcn.posecells
    pc_index = nonzero(pc>.002)
    pc_value = pc[pc_index] * 100
    self.pc_ax.scatter(pc_index[0],pc_index[1],pc_index[2],s=pc_value)
    self.pc_ax.set_xlim3d([0, POSE_SIZE[0]])
    self.pc_ax.set_ylim3d([0, POSE_SIZE[1]])
    self.pc_ax.set_zlim3d([0, POSE_SIZE[2]])
  
  # This is called whenever new visual information is received
  def vis_callback( self, data ):

    cv_im = self.bridge.imgmsg_to_cv( data, "mono8" )
    im = asarray( cv_im )

    pc_max = self.pcn.get_pc_max()
    template_match = self.vts.match( input=im,pc_x=pc_max[0],pc_y=pc_max[1],pc_th=pc_max[2] )
    #TEMP - just inject with this template for now
    self.pcn.inject( .02, template_match.location() )
    
    if DISP_IMAGE:
      self.vt_count += 1
      if self.vt_count > VT_REFRESH_PERIOD:
        self.vt_count = 0
        self.image_data.append( im )
  
  # This is called whenever new odometry information is received
  def odom_callback( self, data ):
    twist = data.twist.twist
    # If there is barely any motion, don't bother flooding the queue with it
    if abs(twist.linear.x) > 0.001 or abs(twist.angular.z) > 0.001:
      self.twist_data.append(twist)

  def run( self ):

    rospy.init_node('posecells', anonymous=True)
    sub_odom = rospy.Subscriber('navbot/odom',Odometry,self.odom_callback)
    sub_vis = rospy.Subscriber('navbot/camera/image',Image,self.vis_callback)
    em_prev_xy = ( 0, 0 )

    while not rospy.is_shutdown():
      if len(self.twist_data) > 0:
        self.em_count += 1
        self.pc_count += 1
        twist = self.twist_data.popleft()
        vtrans = twist.linear.x / self.odom_freq
        vrot = twist.angular.z / self.odom_freq
        self.pcn.update( ( vtrans, vrot ) )
        pc_max = self.pcn.get_pc_max()
        self.em.update( vtrans, vrot, pc_max )
        
        if self.pc_count >= PC_REFRESH_PERIOD:
          self.pc_count = 0
          pc = self.pcn.posecells
          pc_index = nonzero(pc>.002)
          pc_value = pc[pc_index] * 100
          self.pc_ax.scatter(pc_index[0],pc_index[1],pc_index[2],s=pc_value)
          self.pc_ax.set_xlim3d([0, POSE_SIZE[0]])
          self.pc_ax.set_ylim3d([0, POSE_SIZE[1]])
          self.pc_ax.set_zlim3d([0, POSE_SIZE[2]])
        
        if self.em_count >= EM_REFRESH_PERIOD:
          self.em_count = 0
          em_xy = self.em.get_current_point()
          self.em_ax.plot( [ em_prev_xy[0], em_xy[0] ], [ em_prev_xy[1], em_xy[1] ],'b')
          em_prev_xy = em_xy

        plt.pause( 0.0001 ) # This is needed for the display to update
      if len( self.image_data ) > 0:
        im = self.image_data.popleft()
        self.vts_ax.imshow( im, cmap=plt.cm.gray )
        plt.pause( 0.0001 )
        self.im_count += 1
        if self.im_count == NUM_IMAGES:
          break # Exit the loop, so cProfile can return stats

def main():
  ratslam = RatslamRos()
  ratslam.run()

if __name__ == "__main__":
  main()
