from dm_control import mujoco, suite
import wandb
import numpy as np

# XML = '''
# <mujoco>
#   <worldbody>
#     <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
#     <geom type="plane" size="100 100 0.1" rgba=".9 0 0 1"/>
#     <body pos="0 0 1">
#       <joint type="free"/>
#       <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
#     </body>
#   </worldbody>
# </mujoco>
# '''

#wandb.init(project="mujoco-tests", entity="kbganeshan")

# xml_filename = 'ant.xml'
# with open(xml_filename, 'r') as file:
#     XML = file.read()

# physics = mujoco.Physics.from_xml_string(XML)

env = suite.load('swimmer', 'swimmer15')
frames = []
framerate = 30
while env.physics.data.time < 10:
    action = 2 * np.random.rand(*env.action_spec().shape) - 1
    env.step(action)
    while len(frames) < env.physics.data.time * framerate:
        pixels = env.physics.render(camera_id='tracking1')
        frames.append(pixels)
frames = np.moveaxis(frames, -1, 1)
#wandb.log({'video': wandb.Video(frames)})
        
        
    
    


# model = mujoco.MjModel.from_xml_string(XML)
# data = mujoco.MjData(model)
# glfw.init()
# opengl_ctx = glfw.create_window(width=1024, height=728, title='Invisible window', monitor=None, share=None)
# glfw.make_context_current(opengl_ctx)
# ctx = mujoco.MjrContext(model, 10)

# scene = mujoco.MjvScene(model, 1000)
# cam = mujoco.MjvCamera()
# opt = mujoco.MjvOption()
# mujoco.mjv_defaultCamera(cam)
# cam.distance = cam.distance * 2
# mujoco.mjv_defaultOption(opt)
# catmask = mujoco.mjtCatBit.mjCAT_ALL
# count = 0
# while data.time < 10 and not glfw.window_should_close(opengl_ctx):
#     count += 1
#     mujoco.mj_step(model, data)
#     size = glfw.get_framebuffer_size(opengl_ctx)
#     viewport = mujoco.MjrRect(*(0, 0, *size))
#     mujoco.mjv_updateScene(model, data, opt, None, cam, catmask, scene)
#     mujoco.mjr_render(viewport, scene, ctx)
#     glfw.swap_buffers(opengl_ctx)
#     glfw.poll_events()
    
# print(f"done: {count}")