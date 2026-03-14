from crowd_sim.envs.utils.state import FullState, ObservableState, JointState
from crowd_sim.envs.policy.policy_factory import policy_factory
import socket
import struct
import json
import numpy as np
import time
import threading
import os
import sys
from datetime import datetime

# 将项目根目录加入路径，以便导入 crowd_sim / crowd_nav
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# 从 TCP 地图生成静态障碍物时使用 scripts/map_trans（运行时通过 sys.path 解析）
scripts_dir = os.path.join(root_dir, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from map_trans import convert_from_dict  # type: ignore[import-untyped]

# ---------- 日志配置 ----------
LOG_DIR = os.path.join(root_dir, "unity_logs")
LOG_UNITY_DATA = True  # 是否保存Unity发来的数据
LOG_RESPONSE_DATA = True  # 是否保存发送给Unity的响应数据
LOG_MERGE_FREQUENT = True  # 高频消息（如Pedestrian_Update）是否合并到同一文件
LOG_FREQUENT_TYPES = {"Pedestrian_Update"}  # 需要合并的高频消息类型


def ensure_log_dir():
    """确保日志目录存在"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        print(f"[日志] 创建日志目录: {LOG_DIR}")


def _get_date_dir():
    """获取按日期创建的子目录"""
    ensure_log_dir()
    date_str = datetime.now().strftime("%Y-%m-%d")
    date_dir = os.path.join(LOG_DIR, date_str)
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)
    return date_dir


def save_unity_message(msg, addr):
    """将Unity发来的消息保存到本地文件

    - Init_Map 等低频消息：每条单独保存为一个文件
    - Pedestrian_Update 等高频消息：追加到同一个 JSONL 文件中
    """
    if not LOG_UNITY_DATA:
        return

    date_dir = _get_date_dir()
    msg_type = msg.get("msg_type", "unknown")
    timestamp = datetime.now().strftime("%H-%M-%S-%f")

    log_entry = {
        "direction": "recv",
        "timestamp": datetime.now().isoformat(),
        "client_addr": str(addr),
        "message": msg
    }

    try:
        if LOG_MERGE_FREQUENT and msg_type in LOG_FREQUENT_TYPES:
            filename = f"recv_{msg_type}.jsonl"
            filepath = os.path.join(date_dir, filename)
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        else:
            filename = f"recv_{msg_type}_{timestamp}.json"
            filepath = os.path.join(date_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[日志] 保存接收消息失败: {e}")


def save_response_message(msg, addr):
    """将发送给Unity的响应消息保存到本地文件"""
    if not LOG_RESPONSE_DATA:
        return

    date_dir = _get_date_dir()
    msg_type = msg.get("msg_type", "unknown")
    timestamp = datetime.now().strftime("%H-%M-%S-%f")

    log_entry = {
        "direction": "send",
        "timestamp": datetime.now().isoformat(),
        "client_addr": str(addr),
        "message": msg
    }

    try:
        if LOG_MERGE_FREQUENT and msg_type in LOG_FREQUENT_TYPES:
            filename = f"send_{msg_type}.jsonl"
            filepath = os.path.join(date_dir, filename)
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        else:
            filename = f"send_{msg_type}_{timestamp}.json"
            filepath = os.path.join(date_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[日志] 保存发送消息失败: {e}")


# 服务端状态管理
class ServerState:
    def __init__(self):
        self.map_data = None  # 存储地图数据
        self.static_obstacles = []  # 由地图转换得到的静态障碍物（boxes_2d 格式转成的顶点列表）
        self.is_initialized = False
        self.clients = {}  # 保存客户端连接信息


# 服务端实例
server_state = ServerState()

# CentralizedORCA 单例：一次仿真计算所有行人速度，避免 N 次 for 循环带来的延迟
_centralized_orca_policy = None


def static_obstacles_from_boxes_2d(boxes_2d):
    """将 boxes_2d 格式转为 RVO2 使用的静态障碍物列表。
    boxes_2d: [{"vertices": [[x,z], ...]}, ...]（由 map_trans 产生）
    返回: List[List[Tuple[float, float]]]，逆时针顶点。
    """
    obstacles = []
    for item in boxes_2d:
        verts = item.get("vertices", [])
        if len(verts) >= 2:
            obstacles.append([(float(v[0]), float(v[1])) for v in verts])
    return obstacles


def get_centralized_orca_policy():
    """获取或创建 CentralizedORCA 策略，一次 RVO2 仿真得到所有行人速度。
    行人策略与机器人一致：期望速度/接近目标减速/最小爬行/远离他人时加速，仅允许行人后退。
    """
    global _centralized_orca_policy
    if _centralized_orca_policy is None:
        _centralized_orca_policy = policy_factory["centralized_orca"]()
        _centralized_orca_policy.set_phase("test")
        _centralized_orca_policy.set_time_step(0.25)
        _centralized_orca_policy.safety_space = 0.2
        _centralized_orca_policy.static_obstacles = getattr(
            server_state, "static_obstacles", []
        )
        # 与机器人策略一致的参数（仅 allow_backward=True 允许行人后退）
        _centralized_orca_policy.goal_radius = 0.05
        _centralized_orca_policy.approach_radius = 0.5
        _centralized_orca_policy.min_pref_speed = 0.5
        _centralized_orca_policy.min_creep_speed = 0.15
        _centralized_orca_policy.far_from_human_threshold = 3.0
        _centralized_orca_policy.min_speed_when_far_from_humans = 0.6
        _centralized_orca_policy.allow_backward = True
    return _centralized_orca_policy


# ---------- 工具函数 ----------
def recv_all(sock, length):
    """确保从socket接收指定长度的数据"""
    data = b''
    while len(data) < length:
        packet = sock.recv(length - len(data))
        if not packet:
            raise ConnectionError("Socket connection closed")
        data += packet
    return data


def parse_server_message(sock):
    """从socket中接收并解析ServerMsg消息（包括JSON+图像数据）"""

    # 1. 读取 JSON 部分长度（4字节）
    raw_len = recv_all(sock, 4)
    json_len = struct.unpack('<I', raw_len)[0]

    # 2. 读取 JSON 数据
    json_bytes = recv_all(sock, json_len)
    msg_dict = json.loads(json_bytes.decode('utf-8'))
    # print("🔵 收到消息:", msg_dict)
    return msg_dict


def send_client_msg(sock, client_msg, addr=None):
    """将 ClientMsg 结构回发给 Unity"""
    msg_json = json.dumps(client_msg)
    msg_bytes = msg_json.encode('utf-8')

    length = len(msg_bytes)
    # print(f"🔴 回传消息: {msg_json[:100]}... (长度: {length})")
    sock.sendall(struct.pack('<I', len(msg_bytes)) + msg_bytes)

    # 保存发送的消息到日志
    save_response_message(client_msg, addr)


def _vec2(obj, x_key="x", z_key="z"):
    """从协议中的 pos/velocity/target 取 2D 平面 (x, z)，对应 ORCA 的 (px, py)"""
    return (float(obj[x_key]), float(obj[z_key]))


def compute_v_pref_by_density(my_pos, other_positions, *,
                              nearby_radius=4.0,
                              v_max_no_neighbor=1.2,
                              v_min_dense=0.4,
                              count_for_min_speed=5):
    """密度自适应速度：周围智能体/障碍越多，期望速度越低；无人时用 v_max。

    :param my_pos: (x, y) 当前行人位置
    :param other_positions: [(x,y), ...] 其他行人+动态障碍物位置（不含自己）
    :param nearby_radius: 在此距离（米）内计为“周围”
    :param v_max_no_neighbor: 周围无人时的最大期望速度
    :param v_min_dense: 周围人多时的最小期望速度
    :param count_for_min_speed: 达到此人数时使用 v_min_dense，中间线性插值
    :return: 计算得到的 v_pref (float)
    """
    mx, my = float(my_pos[0]), float(my_pos[1])
    r2 = nearby_radius * nearby_radius
    nearby_count = 0
    for pos in other_positions:
        ox, oy = float(pos[0]), float(pos[1])
        if (ox - mx) ** 2 + (oy - my) ** 2 <= r2:
            nearby_count += 1
    if nearby_count == 0:
        return v_max_no_neighbor
    if nearby_count >= count_for_min_speed:
        return v_min_dense
    t = nearby_count / count_for_min_speed
    return v_max_no_neighbor - t * (v_max_no_neighbor - v_min_dense)


def _safe_agent_state(agent):
    """校验并提取单个行人字段，缺省时用默认值，无效则返回 None。"""
    try:
        if "pos" not in agent or "velocity" not in agent or "target" not in agent:
            return None
        pos = _vec2(agent["pos"])
        vel = _vec2(agent["velocity"])
        tgt = _vec2(agent["target"])
        r = float(agent.get("radius", 0.5))
        r = max(0.01, min(r, 2.0))
        v_pref = float(agent.get("v_pref") or agent.get("maxSpeed") or 1.0)
        v_pref = max(0.1, min(v_pref, 2.0))
        return (pos, vel, tgt, r, v_pref, agent.get("id"))
    except (KeyError, TypeError, ValueError):
        return None


def calculate_avoidance(agents, obstacles):
    """
    使用 CentralizedORCA 一次仿真为所有行人计算避障速度。
    - agents: 行人，有 target，需要返回 desiredVelocity。
    - obstacles: 动态障碍物（如机器人），有 pos、velocity、radius；不返回指令，但参与 ORCA 约束，
      使行人能根据障碍物的当前位置与速度进行避让。障碍物的 preferred velocity 设为当前速度，
      以便 RVO2 做互惠避障时考虑其运动趋势。
    协议格式参考 PathPlanningProtocol。
    """
    if not agents:
        return []

    policy = get_centralized_orca_policy()

    # 行人：校验并解析，得到有效列表（含 pos, vel, tgt, r, v_pref_base, id）
    valid_agents = []
    for agent in agents:
        parsed = _safe_agent_state(agent)
        if parsed is None:
            continue
        valid_agents.append(parsed)
    if not valid_agents:
        return []

    # 收集所有“其他实体”位置：其他行人 + 动态障碍物，供密度自适应用
    all_agent_positions = [p for p, _v, _t, _r, _vp, _id in valid_agents]
    obstacle_positions = []
    for obs in obstacles or []:
        try:
            obstacle_positions.append(_vec2(obs["pos"]))
        except (KeyError, TypeError, ValueError):
            continue

    # 密度自适应速度：每个行人根据周围人数/障碍数计算 v_pref，再构造 FullState
    DENSITY_NEARBY_RADIUS = 5.0
    DENSITY_V_MIN = 0.5
    DENSITY_COUNT_FOR_MIN = 5
    agent_full_states = []
    for i, (pos, vel, tgt, r, v_pref_base, aid) in enumerate(valid_agents):
        other_positions = [all_agent_positions[j] for j in range(
            len(valid_agents)) if j != i] + obstacle_positions
        v_pref = compute_v_pref_by_density(
            pos, other_positions,
            nearby_radius=DENSITY_NEARBY_RADIUS,
            v_max_no_neighbor=v_pref_base,
            v_min_dense=DENSITY_V_MIN,
            count_for_min_speed=DENSITY_COUNT_FOR_MIN,
        )
        agent_full_states.append(FullState(
            pos[0], pos[1], vel[0], vel[1], r,
            tgt[0], tgt[1], v_pref, 0.0
        ))

    # 动态障碍物（如机器人）：位置 + 当前速度已参与 ORCA；目标设为 pos+velocity，
    # 使 preferred velocity = 当前速度，行人能预判并避让运动中的障碍物
    obstacle_full_states = []
    for obs in obstacles or []:
        try:
            p = _vec2(obs["pos"])
            v = _vec2(obs["velocity"])
            r = float(obs.get("radius", 0.5))
            gx, gy = p[0] + v[0], p[1] + v[1]
            obstacle_full_states.append(FullState(
                p[0], p[1], v[0], v[1], r, gx, gy, 1.0, 0.0
            ))
        except (KeyError, TypeError, ValueError):
            continue

    state_list = agent_full_states + obstacle_full_states
    try:
        actions = policy.predict(state_list)
    except Exception as e:
        print(f"[避障] predict 异常: {e}")
        return [{"id": p[5], "action": "move", "desiredVelocity": {"x": 0.0, "y": 0.0, "z": 0.0}} for p in valid_agents]

    # 速度平滑：与上一帧速度混合，减轻突变（1.0=不平滑，0.0=完全沿用当前速度）
    # 例如 0.4 表示 40% 新速度 + 60% 当前速度，数值越小越平滑、响应越慢
    VELOCITY_BLEND = 0.8

    commands = []
    for i, parsed in enumerate(valid_agents):
        aid = parsed[5]
        pos, vel, tgt, r, v_pref_base, _ = parsed
        act = actions[i]
        vx = VELOCITY_BLEND * act.vx + (1.0 - VELOCITY_BLEND) * vel[0]
        vz = VELOCITY_BLEND * act.vy + (1.0 - VELOCITY_BLEND) * vel[1]
        commands.append({
            "id": aid,
            "action": "move",
            "desiredVelocity": {"x": vx, "y": 0.0, "z": vz}
        })
    return commands

# ---------- 主服务逻辑 ----------


def handle_client(conn, addr):
    print(f"[连接] 来自 {addr}")

    try:
        while True:
            msg = parse_server_message(conn)

            # 保存Unity发来的数据到本地文件
            save_unity_message(msg, addr)

            if msg["msg_type"] == "Init_Map":
                # 从 TCP 传来的地图信息经 map_trans 转为 boxes_2d 格式，再转为静态障碍物（不读文件）
                server_state.map_data = msg["map"]
                map_data = msg["map"]

                # ---------- 调试：打印地图关键信息 ----------
                print("\n" + "=" * 60)
                print("[地图调试] 收到 Init_Map 消息")
                print(f"  地图名称: {map_data.get('mapName', 'N/A')}")

                # 检查 boxes 数据
                boxes_key = "boxes" if "boxes" in map_data else "Boxs"
                boxes_list = map_data.get(boxes_key, [])
                print(f"  障碍物字段: '{boxes_key}', 数量: {len(boxes_list)}")

                if boxes_list:
                    print(f"  前3个障碍物示例:")
                    for i, box in enumerate(boxes_list[:3]):
                        pos = box.get("position", {})
                        scale = box.get("scale", {})
                        print(f"    [{i}] pos=({pos.get('x', 0):.2f}, {pos.get('z', 0):.2f}), "
                              f"scale=({scale.get('x', 0):.2f}, {scale.get('z', 0):.2f})")
                # ---------- 调试结束 ----------

                if USE_STATIC_OBSTACLES:
                    boxes_2d = convert_from_dict(msg["map"])
                    server_state.static_obstacles = static_obstacles_from_boxes_2d(
                        boxes_2d)

                    # ---------- 调试：打印转换后的静态障碍物 ----------
                    print(
                        f"\n[地图调试] 转换后静态障碍物: {len(server_state.static_obstacles)} 个")
                    if server_state.static_obstacles:
                        print(f"  前3个障碍物顶点:")
                        for i, obs in enumerate(server_state.static_obstacles[:3]):
                            print(
                                f"    [{i}] {len(obs)}个顶点: {obs[0] if obs else 'N/A'} ...")
                    # ---------- 调试结束 ----------
                else:
                    server_state.static_obstacles = []

                policy = get_centralized_orca_policy()
                policy.static_obstacles = server_state.static_obstacles
                if USE_STATIC_OBSTACLES and policy.safety_space > 0.1:
                    policy.safety_space = 0.1
                server_state.is_initialized = True

                # ---------- 调试：确认 policy 设置 ----------
                print(f"\n[地图调试] ORCA策略配置:")
                print(
                    f"  static_obstacles 已设置: {len(policy.static_obstacles)} 个")
                print(f"  safety_space: {policy.safety_space}")
                print(f"  USE_STATIC_OBSTACLES: {USE_STATIC_OBSTACLES}")
                print("=" * 60 + "\n")
                # 返回Init_Map确认
                response = {
                    "msg_type": "Init_Map",
                    "header": {
                        "timestamp": int(time.time() * 1000),
                        "frameId": 0,
                        "mapName": msg["map"]["mapName"],
                        "status": "OK"
                    },
                    "commands": []  # 添加空的commands数组，确保JSON结构完整
                }
                send_client_msg(conn, response, addr)
                print(f"发送Init_Map确认给 {addr}")
            elif msg["msg_type"] == "Pedestrian_Update":
                if "agents" not in msg:
                    msg["agents"] = []
                if "obstacles" not in msg:
                    msg["obstacles"] = []
                policy = get_centralized_orca_policy()
                policy.static_obstacles = getattr(
                    server_state, "static_obstacles", [])

                # ---------- 调试：每100帧打印一次状态 ----------
                frame_id = msg.get("header", {}).get("frameId", 0)
                if frame_id % 100 == 0:
                    print(f"[帧{frame_id}] agents={len(msg['agents'])}, "
                          f"obstacles={len(msg['obstacles'])}, "
                          f"static_obs={len(policy.static_obstacles)}, "
                          f"map_initialized={server_state.is_initialized}")
                # ---------- 调试结束 ----------

                commands = calculate_avoidance(msg["agents"], msg["obstacles"])

                # 7. 构建响应
                response = {
                    "msg_type": "Pedestrian_Update",
                    "header": {
                        "timestamp": int(time.time() * 1000),
                        "frameId": msg["header"]["frameId"],
                        "mapName": msg["header"]["mapName"],
                        "status": "success"
                    },
                    "commands": commands
                }

                # 发送响应
                send_client_msg(conn, response, addr)
            else:
                print(f"错误：期望Init_Map/Pedestrian_Update，收到{msg['msg_type']}")

    except Exception as e:
        print(f"[断开] 客户端异常: {e}")
    finally:
        conn.close()


def start_server(host='10.1.1.95', port=11312):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 防止端口占用
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server_socket.bind((host, port))
    server_socket.listen(5)

    print(f"[启动] 监听地址: {host}:{port}")

    try:
        while True:
            conn, addr = server_socket.accept()

            client_thread = threading.Thread(
                target=handle_client,
                args=(conn, addr),
                daemon=True
            )
            client_thread.start()

    except KeyboardInterrupt:
        print("\n[关闭] 收到 Ctrl+C，正在关闭服务器...")

    finally:
        server_socket.close()
        print("[关闭] Socket 已释放")


# ---------- 启动 ----------
if __name__ == "__main__":
    # 超参数：是否使用静态障碍物（在收到 Init_Map 时由 TCP 地图数据经 map_trans 转为 boxes_2d 后导入）
    USE_STATIC_OBSTACLES = True

    policy = get_centralized_orca_policy()
    policy.static_obstacles = []
    start_server()
