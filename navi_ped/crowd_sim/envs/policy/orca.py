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
        # 只考虑该距离内的行人做避障，超出范围的行人忽略，避免对远处行人过度敏感
        self.neighbor_dist = 3
        self.max_neighbors = 5
        self.time_horizon = 5
        self.time_horizon_obst = 1.5
        self.radius = 0.3
        self.max_speed = 1.5
        self.sim = None
        # 静态障碍物：每项为逆时针顶点列表 [(x,y), ...]，由外部设置（如从 boxes_2d 加载）
        self.static_obstacles = []
        # 行人策略与机器人一致，但允许后退
        self.min_pref_speed = 0.5
        self.goal_radius = 0.05
        self.approach_radius = 0.5
        self.allow_backward = True  # 行人可以后退，其余与机器人策略一致
        self.min_creep_speed = 0.15
        self.far_from_human_threshold = 3.0
        self.min_speed_when_far_from_humans = 0.6
        # 期望速度扰动幅度，用于打破两人相遇时的对称死锁（两人都不动）
        self.perturb_dist = 0.02

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
            self.sim = rvo2.PyRVOSimulator(
                self.time_step, *params, self.radius, self.max_speed)
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

        # Set the preferred velocity to be a vector of unit magnitude (speed) in the direction of the goal.
        velocity = np.array((robot_state.gx - robot_state.px,
                            robot_state.gy - robot_state.py))
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

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
        self.last_state = state

        return action


class CentralizedORCA(ORCA):
    def __init__(self):
        super().__init__()

    def predict(self, state):
        """ Centralized planning for all agents """
        params = self.neighbor_dist, self.max_neighbors, self.time_horizon, self.time_horizon_obst
        obst_count = len(getattr(self, 'static_obstacles', []) or [])
        if self.sim is not None and (
            self.sim.getNumAgents() != len(state)
            or getattr(self, '_static_obstacles_count', -1) != obst_count
        ):
            del self.sim
            self.sim = None
        if self.sim is None:
            self._static_obstacles_count = obst_count
            self.sim = rvo2.PyRVOSimulator(
                self.time_step, *params, self.radius, self.max_speed)
            # 先添加静态障碍物（顶点逆时针），再 processObstacles，最后再添加 agent（RVO2 要求）
            if getattr(self, 'static_obstacles', None):
                for vertices in self.static_obstacles:
                    if len(vertices) >= 2:
                        self.sim.addObstacle(vertices)
                self.sim.processObstacles()
            for agent_state in state:
                self.sim.addAgent(agent_state.position, *params, agent_state.radius + 0.01 + self.safety_space,
                                  self.max_speed, agent_state.velocity)
        else:
            for i, agent_state in enumerate(state):
                self.sim.setAgentPosition(i, agent_state.position)
                self.sim.setAgentVelocity(i, agent_state.velocity)

        # 期望速度：与机器人策略一致，按到目标距离调节（approach_radius / goal_radius / min_pref_speed）
        # 对每个 agent 加小幅扰动，打破两人迎面时的对称死锁（两者都输出 0 速度）
        perturb_dist = getattr(self, 'perturb_dist', 0.02)
        for i, agent_state in enumerate(state):
            to_goal = np.array(
                (agent_state.gx - agent_state.px, agent_state.gy - agent_state.py)
            )
            dist = float(np.linalg.norm(to_goal))
            gr = self.goal_radius
            ar = getattr(self, 'approach_radius', 0.5)
            v_pref = getattr(agent_state, 'v_pref', self.max_speed)
            if dist <= gr:
                pref_vel = (0.0, 0.0)
            else:
                direction = to_goal / dist
                if dist >= ar:
                    mag = v_pref
                else:
                    span = max(ar - gr, 1e-6)
                    t = (dist - gr) / span
                    mag = self.min_pref_speed + \
                        (v_pref - self.min_pref_speed) * t
                    mag = max(self.min_pref_speed, min(v_pref, mag))
                pref_vel = np.array([
                    float(direction[0] * mag),
                    float(direction[1] * mag),
                ])
                # 按 agent 索引加确定性扰动，使不同行人期望速度略有差异，避免对称死锁
                angle = (i * 1.618033988749895) * 2 * np.pi % (2 * np.pi)
                pref_vel[0] += perturb_dist * np.cos(angle)
                pref_vel[1] += perturb_dist * np.sin(angle)
                pref_vel = (float(pref_vel[0]), float(pref_vel[1]))
            self.sim.setAgentPrefVelocity(i, pref_vel)

        self.sim.doStep()

        # 后处理：与机器人策略一致，但 allow_backward=True 故不消除后退分量
        actions = []
        positions = [np.array(agent_state.position, dtype=float)
                     for agent_state in state]
        far_thresh = getattr(self, 'far_from_human_threshold', 3.0)
        min_far = getattr(self, 'min_speed_when_far_from_humans', 0.6)
        deadlock_creep = getattr(self, 'min_creep_speed', 0.15)  # 打破对称死锁时的最小速度

        for i, agent_state in enumerate(state):
            vx, vy = self.sim.getAgentVelocity(i)
            action = ActionXY(vx, vy)
            to_goal = np.array(
                (agent_state.gx - agent_state.px, agent_state.gy - agent_state.py),
                dtype=float,
            )
            dist = float(np.linalg.norm(to_goal))
            my_pos = positions[i]
            nearest_dist = float('inf')
            for j, pos in enumerate(positions):
                if j != i:
                    d = float(np.linalg.norm(pos - my_pos))
                    nearest_dist = min(nearest_dist, d)

            min_creep = getattr(self, 'min_creep_speed', 0.15)
            if dist > 1e-6:
                goal_dir = to_goal / dist
                v = np.array([action.vx, action.vy], dtype=float)
                if not getattr(self, 'allow_backward', True):
                    backward = float(np.dot(v, goal_dir))
                    if backward < -1e-3:
                        v_new = v - backward * goal_dir
                        action = ActionXY(float(v_new[0]), float(v_new[1]))
                        v = np.array([action.vx, action.vy], dtype=float)
            # 最小爬行速度：仅当 RVO2 输出已大致朝向目标时才施加，避免覆盖避障（如遇静态障碍物时 RVO2 会给出小/零速度）
            v = np.array([action.vx, action.vy], dtype=float)
            speed = float(np.linalg.norm(v))
            if dist > self.goal_radius:
                if speed < min_creep:
                    u = to_goal / dist
                    toward = float(np.dot(v, u)) if speed > 1e-6 else 0.0
                    if toward > 0.1:  # 只在明确朝目标走时提速（0.1 m/s），避免覆盖避障
                        action = ActionXY(
                            float(u[0] * min_creep), float(u[1] * min_creep)
                        )
                    # 对称死锁：速度近似为 0 且附近有其他人（非独自被障碍物挡住）时，给带横向分量的速度打破僵局（按索引左右错开，避免两人同时前冲）
                    elif speed < 0.02 and nearest_dist < far_thresh and nearest_dist > 0.5:
                        u = to_goal / dist
                        tangent = np.array([-u[1], u[0]])
                        side = 1 if (i % 2) == 0 else -1
                        nudge = u * (deadlock_creep * 0.6) + \
                            tangent * (side * 0.08)
                        nnorm = float(np.linalg.norm(nudge))
                        if nnorm > 1e-6:
                            nudge = nudge / nnorm * deadlock_creep
                        action = ActionXY(float(nudge[0]), float(nudge[1]))
            # 与最近其他行人距离 > 阈值时，保证朝向目标速度不低于 min_far；同样不覆盖避障
            if dist > self.goal_radius and dist > 1e-6:
                if nearest_dist > far_thresh:
                    v = np.array([action.vx, action.vy], dtype=float)
                    speed = float(np.linalg.norm(v))
                    if speed < min_far:
                        u = to_goal / dist
                        toward = float(np.dot(v, u)) if speed > 1e-6 else 0.0
                        if toward > 0.1:  # 仅当已在朝目标移动时提速，避免撞向静态障碍物
                            action = ActionXY(
                                float(u[0] * min_far), float(u[1] * min_far)
                            )
            actions.append(action)
        return actions
