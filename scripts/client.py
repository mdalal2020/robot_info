import sys
import rospy
from robot_info.srv import *
import numpy as np

def get_jacobian_client():
    rospy.wait_for_service('get_jacobian')
    try:
        get_jacobian = rospy.ServiceProxy('get_jacobian', GetJacobian)
        resp = get_jacobian()
        # return resp1.jacobian
        return np.array([resp.jacobianr1, resp.jacobianr2, resp.jacobianr3, resp.jacobianr4, resp.jacobianr5, resp.jacobianr6])
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

if __name__ == "__main__":
    angles = np.zeros(7)
    print(get_jacobian_client())