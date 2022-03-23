import mujoco

XML = '''
<mujoco>
  <asset>
    <mesh file="gizmo.stl"/>
  </asset>
  <worldbody>
    <body>
      <freejoint/>
      <geom type="mesh" name="gizmo" mesh="gizmo"/>
    </body>
  </worldbody>
</mujoco>
'''
#model = mujoco.mujoco.MjModel.from_xml_string(XML)
#print(model)