import mujoco

XML = '''
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
'''
model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)
ctx = mujoco.GLContext(128, 128)
viewport = mujoco.MjrRect(0, 0, 128, 128)
scene = mujoco.mjv_makescene()
ctx.make_current()
count = 0
while data.time < 1:
    count += 1
    mujoco.mj_step(model, data)
    mujoco.mjr_render(viewport, )