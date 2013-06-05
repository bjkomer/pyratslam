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

POSE_SIZE = ( 50, 50, 20 )
IM_SIZE = ( 256, 256 ) #( 512, 512 )
X_RANGE = ( 64, 192 ) #( 128, 384 )
Y_RANGE = ( 64, 192 ) #( 128, 384 )
X_STEP = 2
Y_STEP = 2
MATCH_THRESHOLD = 10 # the smaller the value, the easier it is to make a match
DISP_IMAGE = False # whether or not visual images will be displayed (they are quite slow)
IM_REFRESH_PERIOD = 15 # how fast the image displayed refreshes, lower number means faster

class RatslamRos():

  def __init__( self ):
    
    self.twist_data = deque()
    self.image_data = deque() #TODO: these should really not be stored anywhere for long
    self.image_count = 0

    self.em = ExperienceMap()

    self.pcn = PoseCellNetwork(shape=POSE_SIZE) 
    self.pose_fig = plt.figure(1)
    self.im_fig = plt.figure(2)
    self.ax = self.pose_fig.add_subplot(111, projection='3d')
    midpoint = (math.floor(POSE_SIZE[0]/2),math.floor(POSE_SIZE[1]/2),math.floor(POSE_SIZE[2]/2))
    self.pcn.inject( 1, midpoint )
    self.ax.set_xlim3d([0, POSE_SIZE[0]])
    self.ax.set_ylim3d([0, POSE_SIZE[1]])
    self.ax.set_zlim3d([0, POSE_SIZE[2]])
    self.ax.hold(False)
    plt.ion()
    plt.show()

    # TODO: put reasonable values here
    self.vts = ViewTemplates( x_range=X_RANGE, y_range=Y_RANGE, 
                              x_step=X_STEP, y_step=Y_STEP, 
                              im_x=IM_SIZE[0], im_y=IM_SIZE[1], 
                              match_threshold=MATCH_THRESHOLD)

    pc = self.pcn.posecells
    pc_index = nonzero(pc>.002)
    pc_value = pc[pc_index] * 100
    self.ax.scatter(pc_index[0],pc_index[1],pc_index[2],s=pc_value)
    self.ax.set_xlim3d([0, POSE_SIZE[0]])
    self.ax.set_ylim3d([0, POSE_SIZE[1]])
    self.ax.set_zlim3d([0, POSE_SIZE[2]])
    self.bridge = CvBridge()
  
  # This is called whenever new visual information is received
  def vis_callback( self, data ):

    #cv_im = bridge.imgmsg_to_cv( data, "passthrough" )
    cv_im = self.bridge.imgmsg_to_cv( data, "mono8" )
    im = asarray( cv_im )

    pc_max = self.pcn.get_pc_max()
    template_match = self.vts.match( input=im,pc_x=pc_max[0],pc_y=pc_max[1],pc_th=pc_max[2] )
    #TEMP - just inject with this template for now
    self.pcn.inject( .02, template_match.location() )
    
    self.image_count += 1
    if self.image_count > IM_REFRESH_PERIOD and DISP_IMAGE:
      self.image_count = 0
      self.image_data.append( im )
  
  # This is called whenever new odometry information is received
  def odom_callback( self, data ):
    twist = data.twist.twist
    # If there is barely any motion, don't bother flooding the queue with it
    if abs(twist.linear.x) > .01 or abs(twist.angular.z) > 0.01:
      self.twist_data.append(twist)

  def run( self ):

    rospy.init_node('posecells', anonymous=True)
    sub_odom = rospy.Subscriber('navbot/odom',Odometry,self.odom_callback)
    #sub_vis = rospy.Subscriber('navbot/camera/image/compressed',CompressedImage,vis_callback)
    #sub_vis = rospy.Subscriber('navbot/camera/image',Image,self.vis_callback,callback_args=self.image_count)
    sub_vis = rospy.Subscriber('navbot/camera/image',Image,self.vis_callback)
   
    while not rospy.is_shutdown():
      if len(self.twist_data) > 0:
        twist = self.twist_data.popleft()
        vtrans = twist.linear.x
        vrot = twist.angular.z
        self.pcn.update( ( vtrans, vrot ) )
        pc_max = self.pcn.get_pc_max()
        self.em.update( vtrans, vrot, pc_max )
        em_xy = self.em.get_current_point() #TODO: display this in a graph
        pc = self.pcn.posecells
        pc_index = nonzero(pc>.002)
        pc_value = pc[pc_index] * 100
        self.ax.scatter(pc_index[0],pc_index[1],pc_index[2],s=pc_value)
        self.ax.set_xlim3d([0, POSE_SIZE[0]])
        self.ax.set_ylim3d([0, POSE_SIZE[1]])
        self.ax.set_zlim3d([0, POSE_SIZE[2]])
        plt.pause( 0.0001 ) # This is needed for the display to update
      if len( self.image_data ) > 0:
        im = self.image_data.popleft()
        plt.imshow( im, cmap=plt.cm.gray )
        plt.pause( 0.0001 )

def main():
  ratslam = RatslamRos()
  ratslam.run()

if __name__ == "__main__":
  main()
