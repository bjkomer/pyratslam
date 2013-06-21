from morse.builder import *

navbot = ATRV()
navbot.translate(x=1.0, z=0.2)
navbot.properties(Object = True, Graspable = False, Label = "NAVBOT")

keyboard = Keyboard()
keyboard.properties(Speed=3.0)
navbot.append(keyboard)

# An odometry sensor to get odometry information
odometry = Odometry()
odometry.frequency(frequency=10)# TEMP - slow down for testing
navbot.append(odometry)
odometry.add_interface('ros', topic="/navbot/odom")

# Add a visual video camera
camera = VideoCamera()
camera.translate(x=0.2, y=0.0, z=0.9)
#camera.properties(cam_far=700, cam_height=512, cam_width=512, Vertical_Flip=True)
#camera.properties(cam_far=700, cam_height=256, cam_width=256, Vertical_Flip=True)
camera.properties(cam_far=900, cam_height=128, cam_width=128, Vertical_Flip=True) #TEMP low quality for fastness
camera.frequency(frequency=10)
navbot.append(camera)
camera.add_interface('ros', topic="/navbot/camera")

# Motion control from an external source (optional)
motion = MotionVW()
navbot.append(motion)
motion.add_interface('ros', topic='/navbot/motion')

env = Environment('/home/komer/Downloads/BUERAKI_v0.2.0/levels/terrains/vastland')
env.place_camera([10.0, -10.0, 10.0])
env.aim_camera([1.0470, 0, 0.7854])
env.select_display_camera(camera)
