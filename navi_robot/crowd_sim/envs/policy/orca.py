import numpy as np
import rvo2
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class ORCA(Policy):
    def __init__(self):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        super().__init__()
        self.name = 'ORCA'
        self.trainable = False
        self.multiagent_training = True
        self.kinematics = 'holonomic'
        self.safety_space = 0
        self.neighbor_dist = 5
        self.max_neighbors = 5
        self.time_horizon = 5
        # 静态障碍物时间视野：越小障碍对速度约束越弱、机器人越不易被“卡慢”，建议 1.5～3.0
        self.time_horizon_obst = 1.5
        self.radius = 0.3
        self.max_speed = 2
        self.sim = None
        # 静态障碍物：每项为逆时针顶点列表 [(x,y), ...]，由外部设置（如从 boxes_2d.json 加载）
        self.static_obstacles = []
        # 接近目标时仍保持的最小期望速度，设高一些保证到达前速度平滑、不陡降
        self.min_pref_speed = 0.5
        # 与目标距离小于此值视为到达，期望速度设为 0
        self.goal_radius = 0.05
        # 进入此半径后才开始减速，之外保持 v_pref；减速区用线性插值到 min_pref_speed，保证平滑
        self.approach_radius = 0.5
        # 是否允许相对目标的后退速度（False 时更符合真人-机器人交互：侧移/停止而非大幅后退）
        self.allow_backward = False
        # 未到目标时若 ORCA 输出速度过小则使用的最小朝向目标速度，避免长期为 0 不前进
        self.min_creep_speed = 0.15
        # 最近行人距离大于此值（米）时，朝向目标的速度不低于 min_speed_when_far_from_humans
        self.far_from_human_threshold = 3.0
        self.min_speed_when_far_from_humans = 0.6

    def configure(self, config):
        return

    def set_phase(self, phase):
        return

    def predict(self, state):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        robot_state = state.robot_state
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        if self.sim is not None and self.sim.getNumAgents() != len(state.human_states) + 1:
            del self.sim
            self.sim = None
        if self.sim is None:
            time_step = self.time_step if self.time_step is not None else 0.25
            self.sim = rvo2.PyRVOSimulator(
                time_step, *params, self.radius, self.max_speed)
            # 先添加静态障碍物（顶点逆时针），再 processObstacles，最后再添加 agent（RVO2 要求）
            if getattr(self, 'static_obstacles', None):
                for vertices in self.static_obstacles:
                    if len(vertices) >= 2:
                        self.sim.addObstacle(vertices)
                self.sim.processObstacles()
            self.sim.addAgent(robot_state.position, *params, robot_state.radius + 0.01 + self.safety_space,
                              robot_state.v_pref, robot_state.velocity)
            for human_state in state.human_states:
                self.sim.addAgent(human_state.position, *params, human_state.radius + 0.01 + self.safety_space,
                                  self.max_speed, human_state.velocity)
        else:
            self.sim.setAgentPosition(0, robot_state.position)
            self.sim.setAgentVelocity(0, robot_state.velocity)
            for i, human_state in enumerate(state.human_states):
                self.sim.setAgentPosition(i + 1, human_state.position)
                self.sim.setAgentVelocity(i + 1, human_state.velocity)

        # 期望速度：指向目标；仅进入 approach_radius 后才减速，路点间保持速度连贯
        to_goal = np.array((robot_state.gx - robot_state.px,
                           robot_state.gy - robot_state.py))
        dist = np.linalg.norm(to_goal)
        if dist <= self.goal_radius:
            pref_vel = (0.0, 0.0)
        else:
            direction = to_goal / dist
            approach_radius = getattr(self, 'approach_radius', 0.5)
            gr = self.goal_radius
            if dist >= approach_radius:
                # 未进入接近区：保持满速
                mag = robot_state.v_pref
            else:
                # 接近区 [goal_radius, approach_radius]：线性由 v_pref 过渡到 min_pref_speed，平滑不陡降
                span = max(approach_radius - gr, 1e-6)
                t = (dist - gr) / span  # 1 在 approach_radius，0 在 goal_radius
                mag = self.min_pref_speed + \
                    (robot_state.v_pref - self.min_pref_speed) * t
                mag = max(self.min_pref_speed, min(robot_state.v_pref, mag))
            pref_vel = tuple((direction * mag).tolist())

        # Perturb a little to avoid deadlocks due to perfect symmetry.
        # perturb_angle = np.random.random() * 2 * np.pi
        # perturb_dist = np.random.random() * 0.01
        # perturb_vel = np.array((np.cos(perturb_angle), np.sin(perturb_angle))) * perturb_dist
        # pref_vel += perturb_vel

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, human_state in enumerate(state.human_states):
            # unknown goal position of other humans
            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        action = ActionXY(*self.sim.getAgentVelocity(0))

        # 后处理：保证未到目标时至少有最小朝向目标速度，避免 ORCA/障碍物导致长期为 0
        to_goal = np.array((robot_state.gx - robot_state.px,
                           robot_state.gy - robot_state.py), dtype=float)
        dist = float(np.linalg.norm(to_goal))
        min_creep = getattr(self, 'min_creep_speed', 0.15)
        if dist > 1e-6:
            goal_dir = to_goal / dist
            v = np.array([action.vx, action.vy], dtype=float)
            if not getattr(self, 'allow_backward', True):
                backward = float(np.dot(v, goal_dir))
                if backward < -1e-3:  # 存在明显后退分量
                    v_new = v - backward * goal_dir
                    action = ActionXY(float(v_new[0]), float(v_new[1]))
                    v = np.array([action.vx, action.vy], dtype=float)
        # 未到目标时若速度过小则强制最小朝向目标速度（ORCA 可行域为空/静态障碍过紧时也会生效）
        if dist > self.goal_radius:
            speed = np.linalg.norm([action.vx, action.vy])
            if speed < min_creep:
                u = to_goal / dist if dist > 1e-6 else np.array([1.0, 0.0])
                action = ActionXY(
                    float(u[0] * min_creep), float(u[1] * min_creep))

        # 最近行人距离 > 3m 时，保证朝向目标的速度不低于一定量，避免无谓减速
        if dist > self.goal_radius and dist > 1e-6:
            rpos = np.array(robot_state.position, dtype=float)
            nearest_human_dist = float('inf')
            for h in state.human_states:
                d = float(np.linalg.norm(
                    np.array(h.position, dtype=float) - rpos))
                nearest_human_dist = min(nearest_human_dist, d)
            far_thresh = getattr(self, 'far_from_human_threshold', 3.0)
            min_far = getattr(self, 'min_speed_when_far_from_humans', 0.6)
            if nearest_human_dist > far_thresh:
                v = np.array([action.vx, action.vy], dtype=float)
                speed = float(np.linalg.norm(v))
                if speed < min_far:
                    u = to_goal / dist
                    action = ActionXY(
                        float(u[0] * min_far), float(u[1] * min_far))

        self.last_state = state
        return action


class CentralizedORCA(ORCA):
    def __init__(self):
        super().__init__()

    def predict(self, state):
        """ Centralized planning for all agents """
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        if self.sim is not None and self.sim.getNumAgents() != len(state):
            del self.sim
            self.sim = None

        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(
                self.time_step, *params, self.radius, self.max_speed)
            for agent_state in state:
                self.sim.addAgent(agent_state.position, *params, agent_state.radius + 0.01 + self.safety_space,
                                  self.max_speed, agent_state.velocity)
        else:
            for i, agent_state in enumerate(state):
                self.sim.setAgentPosition(i, agent_state.position)
                self.sim.setAgentVelocity(i, agent_state.velocity)

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        for i, agent_state in enumerate(state):
            velocity = np.array(
                (agent_state.gx - agent_state.px, agent_state.gy - agent_state.py))
            speed = np.linalg.norm(velocity)
            pref_vel = velocity / speed if speed > 1 else velocity
            self.sim.setAgentPrefVelocity(i, (pref_vel[0], pref_vel[1]))

        self.sim.doStep()
        actions = [ActionXY(*self.sim.getAgentVelocity(i))
                   for i in range(len(state))]

        return actions
