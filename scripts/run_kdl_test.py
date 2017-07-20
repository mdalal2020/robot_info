from robot_info.srv import *
import rospy
import numpy as np
from urdf_parser_py.urdf import URDF
import baxter_interface as bi
from pykdl_utils.kdl_kinematics import KDLKinematics

rospy.init_node('get_robot_pose_jacobian_server')
robot = URDF.from_parameter_server(key='robot_description')

kin = KDLKinematics(robot, 'base', 'right_gripper')
arm = bi.Limb('right')

q = arm.joint_angles()
q = [q['right' + '_s0'], q['right' + '_s1'], q['right' + '_e0'], q['right' + '_e1'], q['right' + '_w0'],
     q['right' + '_w1'], q['right' + '_w2']]

end = 'right_gripper'
pose = kin.forward(q, end)
pose = np.squeeze(np.asarray(pose))
pose = [pose[0][3], pose[1][3], pose[2][3]]

print '\n\n' + str(pose)

print '\n\n' + str(kin.jacobian(q))