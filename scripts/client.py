import sys
import rospy
from robot_info.srv import *
import numpy as np

def get_robot_pose_jacobian_client(name, tip):
    rospy.wait_for_service('get_robot_pose_jacobian')
    try:
        get_robot_pose_jacobian = rospy.ServiceProxy('get_robot_pose_jacobian', getRobotPoseAndJacobian, persistent=True)
        resp = get_robot_pose_jacobian(name, tip)
        return [resp.pose, np.array([resp.jacobianr1, resp.jacobianr2, resp.jacobianr3, resp.jacobianr4, resp.jacobianr5, resp.jacobianr6])]
    except rospy.ServiceException as e:
        print(e)

if __name__ == "__main__":
    print(get_robot_pose_jacobian_client('right', '_hand'))
