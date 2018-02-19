import math
import time
from collections import OrderedDict
import intera_interface as ii
from intera_interface import CHECK_VERSION
import numpy as np
import rospy
from numpy import linalg
from robot_info.srv import *

from joint_space_impedance import PDController
from rllab.envs.base import Env
from rllab.spaces.box import Box

NUM_JOINTS = 7

"""
These are just ball-parks. For more specific specs, either measure them
and/or see http://sdk.rethinkrobotics.com/wiki/Hardware_Specifications.
"""

JOINT_ANGLES_HIGH = np.array([
    1.70167993,
    1.04700017,
    3.0541791,
    2.61797006,
    3.05900002,
    2.09400001,
    3.05899961
])

JOINT_ANGLES_LOW = np.array([
    -1.70167995,
    -2.14700025,
    -3.0541801,
    -0.04995198,
    -3.05900015,
    -1.5708003,
    -3.05899989
])

JOINT_VEL_HIGH = 2*np.ones(7)
JOINT_VEL_LOW = -2*np.ones(7)

#Set these based on how fast you want the robot to move
JOINT_TORQUE_HIGH = 1*np.ones(7)
JOINT_TORQUE_LOW = -1*np.ones(7)

JOINT_VALUE_HIGH = {
    'position': JOINT_ANGLES_HIGH,
    'velocity': JOINT_VEL_HIGH,
    'torque': JOINT_TORQUE_HIGH,
}
JOINT_VALUE_LOW = {
    'position': JOINT_ANGLES_LOW,
    'velocity': JOINT_VEL_LOW,
    'torque': JOINT_TORQUE_LOW,
}

END_EFFECTOR_POS_LOW = [
    0.3404830862298487,
    -1.2633121086809487,
    -0.5698485041484043
]

END_EFFECTOR_POS_HIGH = [
    1.1163239572333106,
    0.003933425621414761,
    0.795699462010194
]
END_EFFECTOR_ANGLE_LOW = -1*np.ones(4)
END_EFFECTOR_ANGLE_HIGH = np.ones(4)

END_EFFECTOR_VALUE_LOW = {
    'position': END_EFFECTOR_POS_LOW,
    'angle': END_EFFECTOR_ANGLE_LOW,
}

END_EFFECTOR_VALUE_HIGH = {
    'position': END_EFFECTOR_POS_HIGH,
    'angle': END_EFFECTOR_ANGLE_HIGH,
}

safety_box_lows = np.array([-0.04304189, -0.43462352, 0.27961519])

safety_box_highs = np.array([ 0.84045825,  0.38408276, 1.8880568 ])

MAX_TORQUES = 0.5 * np.array([8, 7, 6, 5, 4, 3, 2])

joint_names = [
    '_l2',
    '_l3',
    '_l4',
    '_l5',
    '_l6',
    '_hand'
]

#feel free to add more!
experiments=[
    'joint_angle|fixed_angle',
    'joint_angle|varying_angle',
    'end_effector_position|fixed_ee',
    'end_effector_position|varying_ee',
    'end_effector_position_orientation|fixed_ee',
    'end_effector_position_orientation|varying_ee'
]

def safe(raw_function):
    def safe_function(*args, **kwargs):
        try:
            return raw_function(*args, **kwargs)
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    return safe_function


class SawyerEnv(Env):
    def __init__(
            self,
            experiment,
            update_hz=20,
            action_mode='torque',
            safety_box=True,
            reward='huber',
            huber_delta=10,
            safety_force_magnitude=2,
            temp=1.05,
            safe_reset_length=150,
            reward_magnitude=1,
            use_safety_checks=True,
    ):

        rospy.init_node('sawyer_env', anonymous=True)
        self.rate = rospy.Rate(update_hz)

        #defaults:
        self.joint_angle_experiment = False
        self.fixed_angle = False
        self.end_effector_experiment_position = False
        self.end_effector_experiment_total = False
        self.fixed_end_effector = False


        self.use_safety_checks = use_safety_checks
        self.reward_magnitude = reward_magnitude

        if experiment == experiments[0]:
            self.joint_angle_experiment=True
            self.fixed_angle = True
        elif experiment == experiments[1]:
            self.joint_angle_experiment=True
        elif experiment == experiments[2]:
            self.end_effector_experiment_position=True
            self.fixed_end_effector = True
        elif experiment == experiments[3]:
            self.end_effector_experiment_position=True
        elif experiment == experiments[4]:
            self.end_effector_experiment_total=True
            self.fixed_end_effector = True
        elif experiment == experiments[5]:
            self.end_effector_experiment_total = True

        self.safety_box = safety_box
        self.safe_reset_length=safe_reset_length

        if reward == 'MSE':
            self.reward_function = self._MSE_reward
        elif reward == 'huber':
            self.reward_function = self._Huber_reward

        self.huber_delta = huber_delta
        self.safety_force_magnitude = safety_force_magnitude
        self.temp = temp

        self.arm = ii.Limb('right')
        self.arm_joint_names = self.arm.joint_names()

        self.PDController = PDController()

        #create a dictionary whose values are functions that set the appropriate values
        action_mode_dict = {
            'angle': self.arm.set_joint_positions,
            'velocity': self.arm.set_joint_velocities,
            'torque': self.arm.set_joint_torques,
        }

        #create a dictionary whose values are functions that return the appropriate values
        observation_mode_dict = {
            'angle': self._joint_angles,
            'velocity': self.arm.joint_velocities,
            'torque': self.arm.joint_efforts,
        }

        self._set_joint_values = action_mode_dict[action_mode]
        self._get_joint_values = observation_mode_dict

        self._action_space = Box(
            JOINT_VALUE_LOW[action_mode],
            JOINT_VALUE_HIGH[action_mode]
        )

        #set up lows and highs for observation space based on which experiment we are running
        #additionally set up the desired angle as well
        if self.joint_angle_experiment:
            if self.use_angle_parameterization:
                lows = np.hstack((
                    np.cos(JOINT_VALUE_LOW['position']),
                    np.sin(JOINT_VALUE_LOW['position']),
                    JOINT_VALUE_LOW['velocity'],
                    JOINT_VALUE_LOW['torque'],
                    END_EFFECTOR_VALUE_LOW['position'],
                    np.cos(JOINT_VALUE_LOW['position']),
                    np.sin(JOINT_VALUE_LOW['position']),
                ))

                highs = np.hstack((
                    np.cos(JOINT_VALUE_HIGH['position']),
                    np.sin(JOINT_VALUE_HIGH['position']),
                    JOINT_VALUE_HIGH['velocity'],
                    JOINT_VALUE_HIGH['torque'],
                    END_EFFECTOR_VALUE_HIGH['position'],
                    np.cos(JOINT_VALUE_HIGH['position']),
                    np.sin(JOINT_VALUE_HIGH['position']),
                ))
            else:
                lows = np.hstack((
                    JOINT_VALUE_LOW['position'],
                    JOINT_VALUE_LOW['velocity'],
                    JOINT_VALUE_LOW['torque'],
                    END_EFFECTOR_VALUE_LOW['position'],
                    JOINT_VALUE_LOW['position'],
                ))

                highs = np.hstack((
                    JOINT_VALUE_HIGH['position'],
                    JOINT_VALUE_HIGH['velocity'],
                    JOINT_VALUE_HIGH['torque'],
                    END_EFFECTOR_VALUE_HIGH['position'],
                    JOINT_VALUE_HIGH['position'],
                ))

            if self.fixed_angle:
                angles = {
                    'right_j6': 3.23098828125,
                    'right_j5': -2.976708984375,
                    'right_j4': -0.100001953125,
                    'right_j3': 1.59925,
                    'right_j2': -1.6326630859375,
                    'right_j1': -0.3456298828125,
                    'right_j0': 0.0382529296875
                }
                angles = np.array([
                    angles['right_j0'],
                    angles['right_j1'],
                    angles['right_j2'],
                    angles['right_j3'],
                    angles['right_j4'],
                    angles['right_j5'],
                    angles['right_j6']
                ])
                if self.use_angle_wrapping:
                    angles = self._wrap_angles(angles)
                if self.use_angle_parameterization:
                    angles = self.parameterize_angles(angles)
                self.desired = angles
            else:
                self._randomize_desired_angles()

        elif self.end_effector_experiment_position:
            if self.use_angle_parameterization:
                lows = np.hstack((
                    np.cos(JOINT_VALUE_LOW['position']),
                    np.sin(JOINT_VALUE_LOW['position']),
                    JOINT_VALUE_LOW['velocity'],
                    JOINT_VALUE_LOW['torque'],
                    END_EFFECTOR_VALUE_LOW['position'],
                    END_EFFECTOR_VALUE_LOW['position'],
                ))

                highs = np.hstack((
                    np.cos(JOINT_VALUE_HIGH['position']),
                    np.sin(JOINT_VALUE_HIGH['position']),
                    JOINT_VALUE_HIGH['velocity'],
                    JOINT_VALUE_HIGH['torque'],
                    END_EFFECTOR_VALUE_LOW['position'],
                    END_EFFECTOR_VALUE_LOW['position'],
                ))
            else:
                lows = np.hstack((
                    JOINT_VALUE_LOW['position'],
                    JOINT_VALUE_LOW['velocity'],
                    JOINT_VALUE_LOW['torque'],
                    END_EFFECTOR_VALUE_LOW['position'],
                    END_EFFECTOR_VALUE_LOW['position'],
                ))

                highs = np.hstack((
                    JOINT_VALUE_HIGH['position'],
                    JOINT_VALUE_HIGH['velocity'],
                    JOINT_VALUE_HIGH['torque'],
                    END_EFFECTOR_VALUE_HIGH['position'],
                    END_EFFECTOR_VALUE_HIGH['position'],
                ))

            if self.fixed_end_effector:
                self.desired = np.array([0.68998028, -0.2285752, 0.3477])

            else:
                self._randomize_desired_end_effector_pose()

        elif self.end_effector_experiment_total:
            lows = np.hstack((
                JOINT_VALUE_LOW['position'],
                JOINT_VALUE_LOW['velocity'],
                JOINT_VALUE_LOW['torque'],
                END_EFFECTOR_VALUE_LOW['position'],
                END_EFFECTOR_VALUE_LOW['angle'],
                END_EFFECTOR_VALUE_LOW['position'],
                END_EFFECTOR_VALUE_LOW['angle'],
            ))

            highs = np.hstack((
                JOINT_VALUE_HIGH['position'],
                JOINT_VALUE_HIGH['velocity'],
                JOINT_VALUE_HIGH['torque'],
                END_EFFECTOR_VALUE_HIGH['position'],
                END_EFFECTOR_VALUE_HIGH['angle'],
                END_EFFECTOR_VALUE_HIGH['position'],
                END_EFFECTOR_VALUE_HIGH['angle'],
            ))

            if self.fixed_end_effector:
                #hardcoded default, can change to anything you want based on your task
                self.desired = np.array(
                    [0.598038329445, -0.110192662364, 0.273337957845, 0.999390065723, 0.0329420607071, 0.00603632837369,
                     -0.00989342758435])
            else:
                self._randomize_desired_end_effector_pose()

        self._observation_space = Box(lows, highs)
        self._rs = ii.RobotEnable(CHECK_VERSION)
        self.update_pose_and_jacobian_dict()
        self.in_reset = True
        self.amplify=np.ones(7)

    @safe
    def _act(self, action):
        if self.safety_box:
            self.update_pose_and_jacobian_dict()
            self.check_joints_in_box(self.pose_jacobian_dict)
            if len(self.pose_jacobian_dict) > 0:
                forces_dict = self.get_adjustment_forces_per_joint_dict(self.pose_jacobian_dict)
                torques = np.zeros(7)
                for joint in forces_dict:
                    torques = torques + np.dot(self.pose_jacobian_dict[joint][1].T, forces_dict[joint]).T
                if self.remove_action:
                    action = torques
                else:
                    action = action + torques
        if self.in_reset:
            np.clip(action, -4, 4, out=action)
        if not self.in_reset:
            action = self.amplify * action
            action = np.clip(np.asarray(action),-MAX_TORQUES, MAX_TORQUES)
        joint_to_values = dict(zip(self.arm_joint_names, action))
        self._set_joint_values(joint_to_values)
        self.rate.sleep()
        return action

    def is_in_correct_position(self):
        desired_neutral = np.array([
            6.28115601e+00,
            5.10141089e+00,
            6.28014234e+00,
            2.17755176e+00,
            9.48242187e-04,
            5.73669922e-01,
            3.31514160e+00
        ])
        desired_neutral = (desired_neutral)
        actual_neutral = (self._joint_angles())
        errors = self.compute_angle_difference(desired_neutral, actual_neutral)
        ERROR_THRESHOLD = .1*np.ones(7)
        is_within_threshold = (errors < ERROR_THRESHOLD).all()
        return is_within_threshold

    def _wrap_angles(self, angles):
        return angles % (2*np.pi)

    def _joint_angles(self):
        joint_to_angles = self.arm.joint_angles()
        angles =  np.array([
            joint_to_angles[joint] for joint in self.arm_joint_names
        ])
        return angles

    def _end_effector_pose(self):
        state_dict = self.arm.endpoint_pose()
        pos = state_dict['position']
        if self.end_effector_experiment_total:
            orientation = state_dict['orientation']
            return np.array([
                pos.x,
                pos.y,
                pos.z,
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w
            ])
        else:
            return np.array([
                pos.x,
                pos.y,
                pos.z
            ])

    def _MSE_reward(self, differences):
        reward = -np.mean(differences**2)
        return reward

    def _Huber_reward(self, differences):
        a = np.abs(np.mean(differences))
        if a <= self.huber_delta:
            reward = -1 / 2 * a ** 2 * self.reward_magnitude
        else:
            reward = -1 * self.huber_delta * (a - 1 / 2 * self.huber_delta) * self.reward_magnitude
        return reward

    def compute_angle_difference(self, angles1, angles2):
        self._wrap_angles(angles1)
        self._wrap_angles(angles2)
        deltas = np.abs(angles1 - angles2)
        differences = np.minimum(2 * np.pi - deltas, deltas)
        return differences

    def step(self, action, task='reaching'):
        self.nan_check(action)
        actual_commanded_action = self._act(action)
        observation = self._get_observation()
        reward = self.rewards(action, task)

        if self.use_safety_checks:
            out_of_box = self.safety_box_check()
            high_torque = self.high_torque_check(actual_commanded_action)
            unexpected_velocity = self.unexpected_velocity_check()
            unexpected_torque = self.unexpected_torque_check()
            done = out_of_box or high_torque or unexpected_velocity or unexpected_torque
        else:
            done = False
        info = {}
        return observation, reward, done, info

    def rewards(self):
        if self.joint_angle_experiment:
            current = self._joint_angles()
            differences = self.compute_angle_difference(current, self.desired)
            reward = self.reward_function(differences)

        elif self.end_effector_experiment_position or self.end_effector_experiment_total:
            current = self._end_effector_pose()
            # reward = -1*np.linalg.norm(self.desired-current) * self.reward_magnitude
            differences = self.desired-current
            reward = self.reward_function(differences)
        return reward

    def safety_box_check(self):
        self.update_pose_and_jacobian_dict()
        self.check_joints_in_box(self.pose_jacobian_dict)
        terminate_episode = False
        if len(self.pose_jacobian_dict) > 0:
            for joint in self.pose_jacobian_dict.keys():
                dist = self.compute_distances_outside_box(self.pose_jacobian_dict[joint][0])
                if dist > .19:
                    if not self.in_reset:
                        print('safety box failure during train/eval: ', joint, dist)
                        terminate_episode = True
                    else:
                        raise EnvironmentError('safety box failure during reset: ', joint, dist)
        return terminate_episode

    def update_pose_and_jacobian_dict(self):
        self.pose_jacobian_dict = self._get_robot_pose_jacobian_client('right')

    def unexpected_torque_check(self):
        #we care about the torque that was observed to make sure it hasn't gone too high
        new_torques = self.get_observed_torques_minus_gravity()
        if not self.in_reset:
            ERROR_THRESHOLD = np.array([25, 25, 25, 25, 666, 666, 10])
            is_peaks = (np.abs(new_torques) > ERROR_THRESHOLD).any()
            if is_peaks:
                print('unexpected_torque during train/eval: ', new_torques)
                return True
        else:
            ERROR_THRESHOLD = np.array([25, 25, 25, 30, 666, 666, 10])
            is_peaks = (np.abs(new_torques) > ERROR_THRESHOLD).any()
            if is_peaks:
                raise EnvironmentError('unexpected torques during reset: ', new_torques)
        return False


    def unexpected_velocity_check(self):
        velocities_dict = self._get_joint_values['velocity']()
        velocities = np.array([velocities_dict[joint] for joint in self.arm_joint_names])
        ERROR_THRESHOLD = 5 * np.ones(7)
        is_peaks = (np.abs(velocities) > ERROR_THRESHOLD).any()
        if is_peaks:
            print('unexpected_velocities during train/eval: ', velocities)
            if not self.in_reset:
                return True
            else:
                raise EnvironmentError('unexpected velocities during reset: ', velocities)
        return False

    def get_positions_from_pose_jacobian_dict(self):
        poses = []
        for joint in self.pose_jacobian_dict.keys():
            poses.append(self.pose_jacobian_dict[joint][0])
        return np.array(poses)

    def high_torque_check(self, commanded_torques):
        new_torques = np.abs(commanded_torques)
        current_angles = self._joint_angles()
        position_deltas = np.abs(current_angles - self.previous_angles)
        DELTA_THRESHOLD = .05 * np.ones(7)
        ERROR_THRESHOLD = [11, 15, 15, 15, 666, 666, 10]
        violation = False
        for i in range(len(new_torques)):
            if new_torques[i] > ERROR_THRESHOLD[i] and position_deltas[i] < DELTA_THRESHOLD[i]:
                violation=True
                print("violating joint:", i)
        if violation:
            print('high_torque:', new_torques)
            print('positions', position_deltas)
            if not self.in_reset:
                return True
            else:
                raise EnvironmentError('ERROR: Applying large torques and not moving')
        self.previous_angles = current_angles
        return False

    def nan_check(self, action):
        for val in action:
            if math.isnan(val):
                raise EnvironmentError('ERROR: NaN action attempted')

    def _get_observation(self):
        angles = self._joint_angles()
        torques_dict = self._get_joint_values['torque']()
        velocities_dict = self._get_joint_values['velocity']()
        velocities = np.array([velocities_dict[joint] for joint in self.arm_joint_names])
        torques = np.array([torques_dict[joint] for joint in self.arm_joint_names])
        temp = np.hstack((
            angles,
            velocities,
            torques,
            self._end_effector_pose(),
            self.desired
        ))
        return temp

    def safe_move_to_neutral(self):
        for i in range(self.safe_reset_length):
            torques = self.PDController._update_forces()
            actual_commanded_actions = self._act(torques)
            curr_time = time.time()
            self.init_delay = curr_time
            if self.previous_angles_reset_check():
                break
            if self.use_safety_checks:
                self.safety_box_check()
                self.unexpected_torque_check()
                self.high_torque_check(actual_commanded_actions)
                self.unexpected_velocity_check()

    def previous_angles_reset_check(self):
        close_to_desired_reset_pos = self.is_in_correct_position()
        velocities_dict = self._get_joint_values['velocity']()
        velocities = np.abs(np.array([velocities_dict[joint] for joint in self.arm_joint_names]))
        VELOCITY_THRESHOLD = .002 * np.ones(7)
        no_velocity = (velocities < VELOCITY_THRESHOLD).all()
        return close_to_desired_reset_pos and no_velocity

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self.in_reset = True
        self.previous_angles = self._joint_angles()

        if not self.fixed_angle and self.joint_angle_experiment:
            self._randomize_desired_angles()
        elif not self.fixed_end_effector and not self.end_effector_experiment_position and not self.end_effector_experiment_total:
            self._randomize_desired_end_effector_pose()

        self.safe_move_to_neutral()
        self.previous_angles = self._joint_angles()
        self.in_reset = False
        return self._get_observation()

    def _randomize_desired_angles(self):
        self.desired = np.random.rand(1, 7)[0] * 2 - 1

    def _randomize_desired_end_effector_pose(self):
        if self.end_effector_experiment_position:
            self.desired = np.random.uniform(safety_box_lows, safety_box_highs, size=(1, 3))[0]
        else:
            self.desired = np.random.rand(1, 7)[0] * 2 - 1

    def get_pose_jacobian(self, poses, jacobians):
        pose_jacobian_dict = {}
        counter = 0
        pose_counter = 0
        jac_counter = 0
        poses = np.array(poses)
        jacobians = np.array(jacobians)
        for i in range(len(joint_names)):
            pose = poses[pose_counter:pose_counter + 3]
            jacobian = np.array([
                jacobians[jac_counter:jac_counter + 7],
                jacobians[jac_counter + 7:jac_counter + 14],
                jacobians[jac_counter + 14:jac_counter + 21],
            ])
            pose_counter += 3
            jac_counter += 21
            pose_jacobian_dict['right' + joint_names[counter]] = [pose, jacobian]
            counter += 1
        return pose_jacobian_dict

    def _get_robot_pose_jacobian_client(self, name):
        rospy.wait_for_service('get_robot_pose_jacobian')
        try:
            get_robot_pose_jacobian = rospy.ServiceProxy('get_robot_pose_jacobian', getRobotPoseAndJacobian,
                                                         persistent=True)
            resp = get_robot_pose_jacobian(name)
            pose_jac_dict = self.get_pose_jacobian(resp.poses, resp.jacobians)
            return pose_jac_dict
        except rospy.ServiceException as e:
            print(e)

    def check_joints_in_box(self, joint_dict):
        keys_to_remove = []
        for joint in joint_dict.keys():
            if self.is_in_box(joint_dict[joint][0]):
                keys_to_remove.append(joint)
        for key in keys_to_remove:
            del joint_dict[key]
        return joint_dict

    def is_in_box(self, endpoint_pose):
        within_box = [curr_pose > lower_pose and curr_pose < higher_pose
                      for curr_pose, lower_pose, higher_pose
                      in zip(endpoint_pose, safety_box_lows, safety_box_highs)]
        return all(within_box)

    def get_adjustment_forces_per_joint_dict(self, joint_dict):
        forces_dict = {}
        for joint in joint_dict:
            force = self.get_adjustment_force(joint_dict[joint][0])
            forces_dict[joint] = force
        return forces_dict

    def get_adjustment_force(self, endpoint_pose):
        x, y, z = 0, 0, 0

        curr_x = endpoint_pose[0]
        curr_y = endpoint_pose[1]
        curr_z = endpoint_pose[2]
        if curr_x > safety_box_highs[0]:
            x = -1 * np.exp(np.abs(curr_x - safety_box_highs[0]) * self.temp) * self.safety_force_magnitude
        elif curr_x < safety_box_lows[0]:
            x = np.exp(np.abs(curr_x - safety_box_lows[0]) * self.temp) * self.safety_force_magnitude

        if curr_y > safety_box_highs[1]:
            y = -1 * np.exp(np.abs(curr_y - safety_box_highs[1]) * self.temp) * self.safety_force_magnitude
        elif curr_y < safety_box_lows[1]:
            y = np.exp(np.abs(curr_y - safety_box_lows[1]) * self.temp) * self.safety_force_magnitude

        if curr_z > safety_box_highs[2]:
            z = -1 * np.exp(np.abs(curr_z - safety_box_highs[2]) * self.temp) * self.safety_force_magnitude
        elif curr_z < safety_box_lows[2]:
            z = np.exp(np.abs(curr_z - safety_box_highs[2]) * self.temp) * self.safety_force_magnitude
        return np.array([x, y, z])

    def compute_distances_outside_box(self, pose):
        curr_x = pose[0]
        curr_y = pose[1]
        curr_z = pose[2]
        if(self.is_in_box(pose)):
            x, y, z = 0, 0, 0
        else:
            x, y, z = 0, 0, 0
            if curr_x > safety_box_highs[0]:
                x = np.abs(curr_x - safety_box_highs[0])
            elif curr_x < safety_box_lows[0]:
                x = np.abs(curr_x - safety_box_lows[0])
            if curr_y > safety_box_highs[1]:
                y = np.abs(curr_y - safety_box_highs[1])
            elif curr_y < safety_box_lows[1]:
                y = np.abs(curr_y - safety_box_lows[1])
            if curr_z > safety_box_highs[2]:
                z = np.abs(curr_z - safety_box_highs[2])
            elif curr_z < safety_box_lows[2]:
                z = np.abs(curr_z - safety_box_lows[2])
        return np.linalg.norm([x, y, z])

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def render(self):
        pass

    def log_diagnostics(self, paths):
        pass

    @property
    def horizon(self):
        raise NotImplementedError

    def terminate(self):
        self.reset()

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass

    def turn_off_robot(self):
        self._rs.stop()
