import rospy
from std_msgs.msg import Empty

import intera_interface
import numpy as np

class PDController(object):
    """
    Modified PD Controller for Moving to Neutral

    @param robot: the name of the robot to run the pd controller
    @param limb_name: limb on which to run the pd controller

    """
    def __init__(self):

        # control parameters
        self._rate = 1000  # Hz
        self._missed_cmds = 20.0  # Missed cycles before triggering timeout

        # create our limb instance

        self._limb = intera_interface.Limb("right")


        # initialize parameters
        self._springs = dict()
        self._damping = dict()
        self._des_angles = dict()

        # create cuff disable publisher
        cuff_ns = 'robot/limb/' + 'right' + '/suppress_cuff_interaction'
        self._pub_cuff_disable = rospy.Publisher(cuff_ns, Empty, queue_size=1)

        self._des_angles = {'right_j0': 0.298009765625, 'right_j2': -0.350818359375, 'right_j4': 0.0557021484375, 'right_j3': 1.1678642578125, 'right_j1': -1.1768076171875, 'right_j6': 3.2978828125, 'right_j5': 1.3938330078125}

        self.max_stiffness = 20
        self.time_to_maxstiffness = .3  ######### 0.68
        self.t_release = rospy.get_time()

        self._imp_ctrl_is_active = True

        for joint in self._limb.joint_names():
            self._springs[joint] = 30
            self._damping[joint] = 4


    def _set_des_pos(self, des_angles_dict):
        self._des_angles = des_angles_dict

    def adjust_springs(self):
        for joint in list(self._des_angles.keys()):
            t_delta = rospy.get_time() - self.t_release
            if t_delta > 0:
                if t_delta < self.time_to_maxstiffness:
                    self._springs[joint] = t_delta/self.time_to_maxstiffness * self.max_stiffness
                else:
                    self._springs[joint] = self.max_stiffness
            else:
                print("warning t_delta smaller than zero!")

    def _update_forces(self):
        """
        Calculates the current angular difference between the start position
        and the current joint positions applying the joint torque spring forces
        as defined on the dynamic reconfigure server.
        """

        # print self._springs
        self.adjust_springs()

        # disable cuff interaction
        if self._imp_ctrl_is_active:
            self._pub_cuff_disable.publish()

        # create our command dict
        cmd = dict()
        # record current angles/velocities
        cur_pos = self._limb.joint_angles()
        cur_vel = self._limb.joint_velocities()
        # calculate current forces

        for joint in list(self._des_angles.keys()):
            # spring portion
            cmd[joint] = self._springs[joint] * (self._des_angles[joint] -
                                                 cur_pos[joint])
            # damping portion
            cmd[joint] -= self._damping[joint] * cur_vel[joint]

        if self.robot == 'sawyer':
            cmd = np.array(
                [cmd['right_j0'], cmd['right_j1'], cmd['right_j2'], cmd['right_j3'], cmd['right_j4'],
                cmd['right_j5'], cmd['right_j6']])
        else:
            if self._limb_name == "right":
                cmd = np.array(
                    [cmd['right_s0'], cmd['right_s1'], cmd['right_e0'], cmd['right_e1'], cmd['right_w0'],
                     cmd['right_w1'], cmd['right_w2']])
            elif self._limb_name == "left":
                cmd = np.array(
                    [cmd['left_s0'], cmd['left_s1'], cmd['left_e0'], cmd['left_e1'], cmd['left_w0'],
                     cmd['left_w1'], cmd['left_w2']])
        return cmd

