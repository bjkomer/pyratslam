from posecell_network import PoseCellNetwork
from numpy import *
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pylab

POSE_SIZE = (50,50,10)

set_printoptions(threshold='nan')

class RatSLAM (object):

  def __init__( self, data=zeros((20,2)), shape=POSE_SIZE ): #NOTE: random default shape for now
    self.cur_step = 0
    self.pcn = self.init_pcn( shape )
    self.data = data

  def init_pcn( self, shape ):
    # create the pose cell network
    pcn = PoseCellNetwork( shape )

    # start off will all of the energy is the central cell
    midpoint = (math.floor(shape[0]/2),math.floor(shape[1]/2),math.floor(shape[2]/2))
    pcn.inject( 1, midpoint )
    self.current_pose_cell = midpoint

    return pcn

  def step( self ):

    self.current_pose_cell = self.pcn.update(self.data[self.cur_step,:])
    self.cur_step += 1

def main( steps=40 ):

  data = zeros((40,2))
  data[:,0] = 3
  data[4:9,1] = pi/4
  #data[:,1] = .3

  print (data)

  sim = RatSLAM(data=data,shape=POSE_SIZE) 

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlim3d([0, POSE_SIZE[0]])
  ax.set_ylim3d([0, POSE_SIZE[1]])
  ax.set_zlim3d([0, POSE_SIZE[2]])
  ax.hold(False)
  plt.ion()
  plt.show()

  for s in xrange( steps ):
    #print ("Step: ",s)
    sim.step()

    pc = sim.pcn.posecells
    pc_index = nonzero(pc>.002)
    pc_value = pc[pc_index] * 100
    ax.scatter(pc_index[0],pc_index[1],pc_index[2],s=pc_value)
    ax.set_xlim3d([0, POSE_SIZE[0]])
    ax.set_ylim3d([0, POSE_SIZE[1]])
    ax.set_zlim3d([0, POSE_SIZE[2]])
    plt.pause(.01)

  
# TODO: maybe put this elsewhere
if __name__ == "__main__":
  main()
