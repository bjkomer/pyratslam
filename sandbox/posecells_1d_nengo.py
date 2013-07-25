import nef
import math

# Global Constants

CELL_DIM = 1 # how much real space each cell represents ( e.g. 2 = 2m per cell )
SIGMA_E = 1 # sigma value for excitatory gaussian component of filter
SIGMA_I = 2 # sigma value for inhibitory gaussian component of filter

# Number of Populations

HEAD_DIR_SIZE = 10
POSE_X_SIZE = 10
FILTER_SIZE = 5

# Number of Neurons

N_POSE_NEURONS = 30
N_POSE_INTER_NEURONS = 100 # Intermediate layer of pose network
N_FILTER_NEURONS = 100 #30
N_HEAD_DIR_NEURONS = 30
N_VELOCITY_NEURONS = 100

# Population Radii

FILTER_POP_RADIUS = 1

net = nef.Network( '1D Pose Network', seed=13 )

net.make_input( 'velocity_input', [0] )
net.make( 'velocity', N_VELOCITY_NEURONS, dimensions=1 )
net.connect( 'velocity_input', 'velocity' )

for i in xrange(POSE_X_SIZE):
  net.make( 'pose_' + str(i), N_POSE_NEURONS, dimensions=1 )

for i in xrange(HEAD_DIR_SIZE):
  net.make( 'head_dir_' + str(i), N_HEAD_DIR_NEURONS, dimensions=1 )

# generate a different function to use for each filter from the velocity input
# basically it will activate the filter that lines up with the velocity the most
# and the others will have a gaussian-esque falling off of activation
def vel_to_filter( index ):

  def func( x ):
    gauss = 1 / (math.sqrt(2*math.pi)) * \
        ( math.exp( -1/2 *( ( x[0] / CELL_DIM - \
            ( index - math.floor( FILTER_SIZE / 2 ) ) ) / SIGMA_E ) ** 2 ) / SIGMA_E - \
          math.exp( -1/2 *( ( x[0] / CELL_DIM - \
            ( index - math.floor( FILTER_SIZE / 2 ) ) ) / SIGMA_I ) ** 2 ) / SIGMA_I )
    
    return 5 * gauss # TODO: use logs later
    if gauss < 0:
      return -FILTER_POP_RADIUS # if negative, hammer it as low as it can go
    else:
      return math.log( gauss, 1.1 ) # use logarithm so multiplication is addition
  
  func.__name__ = 'v2f_' + str( index ) # Nengo cannot have more than one function with the same name
  return func

for i in xrange(FILTER_SIZE):
  net.make( 'filter_' + str(i), N_FILTER_NEURONS, dimensions=1, radius=FILTER_POP_RADIUS )
  net.connect( 'velocity', 'filter_' + str(i), func=vel_to_filter( i ) )

net.make_input( 'inject', [0] ) # Input to inject activity in the system, to create an inital condition
net.connect( 'inject', 'pose_4' )

# Logarithm Method
"""
def exp( x ):
  return math.exp( x[0] )
  #return x[0] # TEMP keeping in log form

for i in xrange(POSE_X_SIZE):
  net.make( 'pose_inter_' + str(i), N_POSE_INTER_NEURONS, dimensions=1 )
  for f in xrange(FILTER_SIZE):
    net.connect( 'pose_' + str( int( ( i + f - math.floor( FILTER_SIZE / 2 ) ) % POSE_X_SIZE ) ), 
                 'pose_inter_' + str(i) )
    net.connect( 'filter_' + str(f), 'pose_inter_' + str(i) )
  net.connect( 'pose_inter_' + str(i), 'pose_' + str(i), func=exp, pstc=1 )
"""

# Multiplication Method

# Function to implement convolution
def conv( x ):
  total = 0
  for i in xrange(FILTER_SIZE):
    total += x[ i ] * x[ i + FILTER_SIZE ]
  return total * 5

for i in xrange(POSE_X_SIZE):
  # Create intermediate population
  net.make( 'pose_inter_' + str(i), N_POSE_INTER_NEURONS * FILTER_SIZE, dimensions=FILTER_SIZE * 2 )
  
  # Build connections to intermediate populations
  for f in xrange(FILTER_SIZE):
    net.connect( 'pose_' + str( int( ( i + f - math.floor( FILTER_SIZE / 2 ) ) % POSE_X_SIZE ) ), 
                 'pose_inter_' + str(i), index_post=[f], pstc=0.001 )
    net.connect( 'filter_' + str(f), 'pose_inter_' + str(i), index_post=[ f + FILTER_SIZE ], pstc=0.001 )
  
  # Build connections from intermediate populations to the pose network
  net.connect( 'pose_inter_' + str(i), 'pose_' + str(i), func=conv, pstc=1 )




"""
f0 = vel_to_filter( 0 )
f1 = vel_to_filter( 1 )
f2 = vel_to_filter( 2 )
f3 = vel_to_filter( 3 )
f4 = vel_to_filter( 4 )

print('f0')
for i in xrange(-16,16):
  print (f0([i]))
print('f1')
for i in xrange(-16,16):
  print (f1([i]))
print('f2')
for i in xrange(-16,16):
  print (f2([i]))
print('f3')
for i in xrange(-16,16):
  print (f3([i]))
print('f4')
for i in xrange(-16,16):
  print (f4([i]))
"""
net.view()
net.add_to_nengo()
