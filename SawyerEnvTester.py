from railrl.envs.ros.sawyer_env import SawyerEnv
import numpy as np
import intera_interface

experiments=[
    'joint_angle|fixed_angle',
    'joint_angle|varying_angle',
    'end_effector_position|fixed_ee',
    'end_effector_position|varying_ee',
    'end_effector_position_orientation|fixed_ee',
    'end_effector_position_orientation|varying_ee'
]

env = SawyerEnv(experiments[0])

env.reset()

env.act(np.ones(7))