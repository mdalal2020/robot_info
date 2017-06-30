import rospy
from baxter_pykdl import baxter_kinematics

rospy.init_node('baxter_kinematics', anonymous=True)
kin = baxter_kinematics('right')
# print kin.jacobian()

# print kin.jacobian()[0][0][0]
import ipdb
ipdb.set_trace()
# print kin.jacobian_transpose()