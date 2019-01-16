import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4


        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])


    def normal(self, x):
        return 1.0/(1.0 + np.sqrt(abs(x)))

    def sigmoid(self, x):
        return 1.0/(1+np.exp(-x))

    def get_my_reward(self):
        """Uses current pose of sim to return reward."""
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        distance = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        ddist = np.tanh(0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        reward = 0.5*self.normal(distance) + 0.5*self.normal(distance)
        #reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        return reward

    def get_reward(self, last_pose):
        """Uses current pose of sim to return reward."""
        max_dist = np.linalg.norm(self.sim.init_pose[:3] - self.target_pos)
        last_dist = np.linalg.norm(last_pose[:3] - self.target_pos)
        cur_dist  = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
        offset_axis_base = self.target_pos - self.sim.pose[:3]
        offset_axis = self.sim.pose[:3] - last_pose[:3]
        if (cur_dist < last_dist):
            flag = 1
        else:
            if offset_axis[2] > 0 and offset_axis_base[2] <=0: # 从下往上爬升
                flag = 1
            elif offset_axis[2] < 0 and offset_axis_base[2] >0: # 下降 高于目标
                flag = 1
            elif offset_axis[2] > 0 and offset_axis_base[2] >0: # 爬升 高于目标
                flag =  -1
            elif offset_axis[2] < 0 and offset_axis_base[2] <0: # 下降 低于目标
                flag =  -1
            else:
                flag =  1
        dist = self.normal(cur_dist)*flag
        #offset = self.normal((1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum())
        #reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        return dist

    def get_reward2(self, last_pose):
        reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        offset = self.normal((1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum())
        return offset

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        last_pose = self.sim.init_pose
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds)
            reward += self.get_reward(last_pose)
            pose_all.append(self.sim.pose)
            last_dist = np.linalg.norm(last_pose[:3] - self.target_pos)
            cur_dist  = np.linalg.norm(self.sim.pose[:3] - self.target_pos)
            if done and (cur_dist < last_dist or self.sim.pose[2] > last_pose[2]):
                reward += 10
            last_pose = self.sim.pose
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state