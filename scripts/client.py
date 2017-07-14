import sys
import rospy
from robot_info.srv import *
import numpy as np

def get_jacobian_client(name):
    rospy.wait_for_service('get_jacobian')
    try:
        get_jacobian = rospy.ServiceProxy('get_jacobian', GetJacobian, persistent=True)
        resp = get_jacobian(name)
        return np.array([resp.jacobianr1, resp.jacobianr2, resp.jacobianr3, resp.jacobianr4, resp.jacobianr5, resp.jacobianr6])
    except rospy.ServiceException as e:
        return resp

if __name__ == "__main__":
    for i in range(10000):
        # print(get_jacobian_client('left'))
        resp = get_jacobian_client('left')
        print(i)
