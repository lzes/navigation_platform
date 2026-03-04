from crowd_sim.envs.utils.state import FullState, ObservableState, JointState
from crowd_nav.policy.policy_factory import policy_factory
import socket
import threading
import importlib.util
import os
import json
import torch
import sys


root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


def receive_data(sock):
    """Receive robot and pedestrian state data sent from the server."""
    data = sock.recv(1024).decode('utf-8')
    return data.split('&')  # Split by '&'


def send_data(sock, vx, vy):
    """Send the robot's computed velocity."""
    msg = f"{vx}, {vy}"
    sock.sendall(msg.encode('utf-8'))
    print(f'send message: {msg}')


def parse_received_data(data):
    """Parse the received data.

    Expected format:
    robot_pos_x,robot_pos_y & robot_vel_x,robot_vel_y & robot_target_x,robot_target_y &
    people1_pos_x,people1_pos_y & people1_vel_x,people1_vel_y & ...
    """
    # Split the raw data by '&'
    fields = data.split('&')

    # Parse robot position, velocity, and target
    robot_pos = list(map(float, fields[0].split(',')))  # Robot position
    robot_vel = list(map(float, fields[1].split(',')))  # Robot velocity
    robot_target = list(map(float, fields[2].split(',')))  # Target position

    # Parse pedestrian data
    humans = []
    for i in range(3, len(fields), 2):
        human_pos = list(map(float, fields[i].split(',')))  # 行人位置
        human_vel = list(map(float, fields[i+1].split(',')))  # 行人速度
        humans.append((human_pos, human_vel))

    return robot_pos, robot_vel, robot_target, humans


def compute_speed_by_pedestrian_count(robot_pos, humans, *,
                                      nearby_radius=4.0,
                                      v_max_no_pedestrian=1.2,
                                      v_min_many_pedestrian=0.4,
                                      count_for_min_speed=5):
    """根据机器人周围行人数量计算期望速度：无行人时加速，行人多时减速。

    :param robot_pos: [x, y] 机器人位置
    :param humans: [(human_pos, human_vel), ...] 行人列表
    :param nearby_radius: 在此距离（米）内计为“周围行人”
    :param v_max_no_pedestrian: 周围无行人时的最大期望速度
    :param v_min_many_pedestrian: 行人很多时的最小期望速度
    :param count_for_min_speed: 达到此人数时使用 v_min_many_pedestrian，中间线性插值
    :return: 计算得到的 v_pref (float)
    """
    rx, ry = float(robot_pos[0]), float(robot_pos[1])
    nearby_count = 0
    for human_pos, _ in humans:
        hx, hy = float(human_pos[0]), float(human_pos[1])
        dist_sq = (hx - rx) ** 2 + (hy - ry) ** 2
        if dist_sq <= nearby_radius * nearby_radius:
            nearby_count += 1
    if nearby_count == 0:
        return v_max_no_pedestrian
    if nearby_count >= count_for_min_speed:
        return v_min_many_pedestrian
    # 线性插值：人数从 0 到 count_for_min_speed，速度从 v_max 降到 v_min
    t = nearby_count / count_for_min_speed
    return v_max_no_pedestrian - t * (v_max_no_pedestrian - v_min_many_pedestrian)


def load_static_obstacles_from_boxes_2d(json_path=None):
    """从 boxes_2d.json 加载静态障碍物（逆时针顶点）。
    格式: [{"vertices": [[x,y], ...]}, ...]
    返回: List[List[Tuple[float, float]]]，供 RVO2 addObstacle 使用。
    """
    if json_path is None:
        json_path = os.path.join(root_dir, 'boxes_2d.json')
    if not os.path.isfile(json_path):
        return []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    obstacles = []
    for item in data:
        verts = item.get('vertices', [])
        if len(verts) >= 2:
            obstacles.append([(float(v[0]), float(v[1])) for v in verts])
    return obstacles


def handle_client_connection(conn, addr):
    try:
        print(f"Connection from {addr}")
        while True:
            # Receive data
            data = conn.recv(1024).decode('utf-8')
            if not data:
                # No data means the client has closed the connection
                break

            # Example of received data:
            # -2.130,-8.880&0.000,0.000&1.900,10.490&-0.081,1.410&-0.007,1.000&-3.359,-3.293&1.000,-0.009
            #
            # Use '&' to split segments, then use ',' to split each segment into
            # X/Z float values (each float with 3 decimal places):
            # -2.130,-8.880 & 0.000,0.000 & 1.900,10.490 & -0.081,1.410 & -0.007,1.000 & -3.359,-3.293 & 1.000,-0.009
            # robot pos     & velocity    &    target    & people1 pos  & people1 vel  & people2 pos   & people2 vel

            print(f"Received data: {data}")

            robot_pos, robot_vel, robot_target, humans = parse_received_data(
                data)

            # 已到达目标：与目标距离小于阈值时直接回 0 速度，避免“接近但未到”时 ORCA 输出 0 导致卡住
            goal_threshold = getattr(policy, 'goal_radius', 0.2)
            dist_to_goal = ((robot_target[0] - robot_pos[0]) **
                            2 + (robot_target[1] - robot_pos[1]) ** 2) ** 0.5
            if dist_to_goal < goal_threshold:
                conn.sendall(b'0.0,0.0')
                continue

            # 根据周围行人数量自动调整期望速度：无行人加速，行人多减速
            v_pref = compute_speed_by_pedestrian_count(robot_pos, humans)
            # 机器人用 v_pref；行人仍用至少 1.0 的 max_speed，避免误限制行人
            policy.max_speed = max(v_pref, 1.0)
            policy.sim = None  # 强制本步用当前 v_pref 重建仿真，使机器人速度限制生效

            # Build the robot state (given robot position, velocity, and target)
            robot_state = FullState(robot_pos[0], robot_pos[1], robot_vel[0],
                                    robot_vel[1], 0.3, robot_target[0], robot_target[1], v_pref, 1.57)

            # Build the pedestrians' states (possibly multiple humans)
            human_states = []
            for human_pos, human_vel in humans:
                human_state = ObservableState(
                    human_pos[0], human_pos[1], human_vel[0], human_vel[1], 0.5)
                human_states.append(human_state)
            print(
                f'human_states:{human_states}  length of human states:{len(human_states)}  v_pref:{v_pref:.2f}')

            # Step 2: Feed into the path planning algorithm
            joint_state = JointState(robot_state, human_states)
            action = policy.predict(joint_state)

            # Step 3: Send the robot velocity
            send_data = f'{action.vx},{action.vy}'
            # send_data = f'{robot_target[0]},{robot_target[1]}'
            conn.sendall(send_data.encode('utf-8'))

    except Exception as e:
        print(f"Exception while handling client {addr}: {e}")
    finally:
        # Close the connection
        conn.close()
        print(f"连接 {addr} 已关闭")


def start_server(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()
        print(f"Server listening on {host}:{port}...")

        try:
            while True:
                # Accept new connection
                conn, addr = s.accept()
                # Create a new thread to handle the TCP connection
                client_thread = threading.Thread(
                    target=handle_client_connection, args=(conn, addr))
                client_thread.start()
        except Exception as e:
            print(f"Server exception: {e}")
        except KeyboardInterrupt:
            print("Server is shutting down...")


if __name__ == '__main__':
    # Configure as needed
    gpu = '0'
    # 与障碍物/行人保持的额外间距；启用静态障碍物时可适当减小（如 0.1）以减轻对速度的抑制
    safety_space = 0.2

    # Configure environment (usually no need to change)
    model_dir = os.path.join(root_dir, 'crowd_nav', 'data', 'output')
    device = torch.device(
        f"cuda:{gpu}" if torch.cuda.is_available() and gpu else "cpu")
    config_file = os.path.join(model_dir, 'config.py')
    model_weights = os.path.join(model_dir, 'best_val.pth')

    print('Loaded RL weights with best VAL')
    spec = importlib.util.spec_from_file_location('config', config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    policy_config = config.PolicyConfig('False')
    # policy = policy_factory[policy_config.name]()
    policy = policy_factory['orca']()
    policy.configure(policy_config)
    # policy.load_model(model_weights)
    train_config = config.TrainConfig('False')
    policy.set_phase('test')
    policy.set_device(device)
    policy.safety_space = safety_space

    # Time step is 0.25s; world coordinate frame; GPU memory usage ~250MB/model
    policy.set_time_step(0.25)

    # 加载静态障碍物（boxes_2d.json：逆时针顶点）
    USE_STATIC_OBSTACLES = True
    policy.static_obstacles = load_static_obstacles_from_boxes_2d(
    ) if USE_STATIC_OBSTACLES else []
    # 启用静态障碍物时可用较小 safety_space 减轻对速度影响（ORCA 内已调小 time_horizon_obst）
    if USE_STATIC_OBSTACLES and safety_space > 0.1:
        safety_space = 0.1
        policy.safety_space = safety_space
    print(
        f'Loaded {len(policy.static_obstacles)} static obstacle(s) from boxes_2d.json')

    # Set server address and port
    HOST = '0.0.0.0'
    # HOST = '127.0.0.1'
    PORT = 11311

    # Start the server
    start_server(HOST, PORT)
