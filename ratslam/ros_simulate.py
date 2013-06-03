# Reads ROS Pose messages and uses these as input to the pose network
from posecell_network import PoseCellNetwork
from numpy import *
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pylab
import rospy
from geometry_msgs.msg import Twist, TwistWithCovariance
from nav_msgs.msg import Odometry
from collections import deque

POSE_SIZE = (50,50,20)

def main():

  twist_data = deque()
  pcn = PoseCellNetwork(shape=POSE_SIZE) 
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  midpoint = (math.floor(POSE_SIZE[0]/2),math.floor(POSE_SIZE[1]/2),math.floor(POSE_SIZE[2]/2))
  pcn.inject( 1, midpoint )
  ax.set_xlim3d([0, POSE_SIZE[0]])
  ax.set_ylim3d([0, POSE_SIZE[1]])
  ax.set_zlim3d([0, POSE_SIZE[2]])
  ax.hold(False)
  plt.ion()
  plt.show()

  pc = pcn.posecells
  pc_index = nonzero(pc>.002)
  pc_value = pc[pc_index] * 100
  ax.scatter(pc_index[0],pc_index[1],pc_index[2],s=pc_value)
  ax.set_xlim3d([0, POSE_SIZE[0]])
  ax.set_ylim3d([0, POSE_SIZE[1]])
  ax.set_zlim3d([0, POSE_SIZE[2]])
  
  
  
  def callback(data):
    twist = data.twist.twist
    # If there is barely any motion, don't bother flooding the queue with it
    if abs(twist.linear.x) > .01 or abs(twist.angular.z) > 0.01:
      twist_data.append(twist)

  rospy.init_node('posecells', anonymous=True)
  sub = rospy.Subscriber('navbot/odom',Odometry,callback)
 
  while not rospy.is_shutdown():
    if len(twist_data) > 0:
      twist = twist_data.popleft()
      vtrans = twist.linear.x
      vrot = twist.angular.z
      pcn.update((vtrans,vrot))
      pc = pcn.posecells
      pc_index = nonzero(pc>.002)
      pc_value = pc[pc_index] * 100
      ax.scatter(pc_index[0],pc_index[1],pc_index[2],s=pc_value)
      ax.set_xlim3d([0, POSE_SIZE[0]])
      ax.set_ylim3d([0, POSE_SIZE[1]])
      ax.set_zlim3d([0, POSE_SIZE[2]])
      plt.pause(.0001)
      
if __name__ == "__main__":
  main()
