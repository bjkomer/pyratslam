from posecell_network import PoseCellNetwork
from numpy import *
from scipy import ndimage
import glumpy
#from matplotlib.pyplot import subplots, close, subplot, gcf
from matplotlib.pyplot import *
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

POSE_SIZE = (10,10,10)

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

def draw_pose( x,y,th, subplot ):
  ax3 = Axes3D(gcf(), rect=subplot.get_position())
  ax3.scatter(x, y, th, 'z')
  ax3.set_xlim3d([0, POSE_SIZE[0]])
  ax3.set_xlim3d([0, POSE_SIZE[1]])
  ax3.set_xlim3d([0, POSE_SIZE[2]])

def main( steps=20 ):

  #from OpenGL import GLUT as glut
  #import time
  import mayavi.mlab

  data = zeros((20,2))
  data[:,0] = 3
  data[4:7,1] = .4
  data[:,1] = .3

  print (data)

  sim = RatSLAM(data=data,shape=POSE_SIZE) 
  
  x = []
  y = []
  th = []

  for s in xrange( steps ):
    #print ("Step: ",s)
    sim.step()

    xp = zeros(POSE_SIZE[0]*POSE_SIZE[1]*POSE_SIZE[2])
    yp = zeros(POSE_SIZE[0]*POSE_SIZE[1]*POSE_SIZE[2])
    zp = zeros(POSE_SIZE[0]*POSE_SIZE[1]*POSE_SIZE[2])
    vp = zeros(POSE_SIZE[0]*POSE_SIZE[1]*POSE_SIZE[2])
    pos = 0
    pc = sim.pcn.posecells
    for i in xrange(POSE_SIZE[0]):
      for j in xrange(POSE_SIZE[1]):
        for k in xrange(POSE_SIZE[2]):
          xp[pos] = i
          yp[pos] = j
          zp[pos] = k
          vp[pos] = pc[i,j,k]
          pos+=1

    pcdata = subplot(2,2,3)
    ax3 = Axes3D(gcf(), rect=pcdata.get_position())
    ax3.scatter(xp,yp,zp,s=vp)
    #mayavi.mlab.points3d(xp, yp, zp, vp)

    #print("")
    """
    im = sim.pcn.posecells[:,:,2]
    imshow(im, cmap=cm.gray)

    pcdata = subplot(2,2,3)
    pc = sim.current_pose_cell
    x.append(pc[0])
    y.append(pc[1])
    th.append(pc[2])
    draw_pose(x,y,th, pcdata)
    title("Pose Cell Activity")
    pcdata.axis('off')
    """

  show()

  """
  
  fig = glumpy.figure((512,512))
  pc = sim.pcn.posecells[:,:,2].astype(float32)
  im = glumpy.image.Image(pc,colormap=glumpy.colormap.Grey)

  @fig.event
  def on_draw():
    fig.clear()
    im.update()
    im.draw( x=0,y=0,z=0,width=fig.width,height=fig.height)
  
  for s in xrange( steps ):
    print ("Step: ",s)
    sim.step()
    print("")
    pc = sim.pcn.posecells[:,:,2].astype(float32)
    im = glumpy.image.Image(pc,colormap=glumpy.colormap.Grey)
    glut.glutMainLoopEvent()
    on_draw()
    glut.glutSwapBuffers()
    time.sleep(1)
  
  """
  
# TODO: maybe put this elsewhere
if __name__ == "__main__":
  main()

"""
# arbitrary testing

pcn = PoseCellNetwork((3,3,3))
pcn.update( 0, 0 )

sigma=1
dim = 7
fil1 = zeros( ( dim, dim ) )
center = math.floor( dim / 2 )
for x in xrange( dim ):
  for y in xrange( dim ):
    fil1[x,y] = 1.0 / (sigma*math.sqrt(2*pi)) * \
        math.exp( (-(x-center)**2 - (y-center)**2 ) / (2*sigma**2))

fil1 /= abs(sum(fil1.ravel())) # normalize

test_in = arange(25).reshape(5,5)
test_in = zeros((5,5))
test_in[2,2] = 1
test_out1 = zeros((5,5))
test_out2 = zeros((5,5))

print( fil1 )

ndimage.gaussian_filter(input = test_in, sigma=sigma, output=test_out2, mode='wrap')
test_out1 = ndimage.correlate(test_in, fil1, mode='wrap',origin=(.5,.5))

print ("with kernel:\n")
print( test_out1 )
print ("with gaussian function:\n")
print( test_out2 )
"""
