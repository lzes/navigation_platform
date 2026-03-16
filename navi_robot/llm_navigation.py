#!/usr/bin/env python3
"""通过 LLM（智谱 GLM）理解自然语言导航指令，从 boxes_2d.json 中检索起点和终点坐标。

示例用法：
    python llm_navigation.py "从货架到收银台"
    
环境变量配置（.env 文件）：
    LLM_API_KEY = "your-api-key"
    LLM_BASE_URL = "https://api-inference.modelscope.cn/v1/"
    LLM_MODEL_ID = "ZhipuAI/GLM-5"
"""

import argparse
import json
import os
import re
from typing import Optional, Tuple, List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

root_dir = os.path.dirname(os.path.abspath(__file__))

# 从环境变量读取配置
API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://api-inference.modelscope.cn/v1/")
MODEL_ID = os.getenv("LLM_MODEL_ID", "ZhipuAI/GLM-5")


class ZhipuLLMClient:
    """智谱 GLM 大语言模型客户端（兼容 OpenAI 接口）。"""
    
    def __init__(self, model: str = None, api_key: str = None, base_url: str = None):
        self.model = model or MODEL_ID
        self.api_key = api_key or API_KEY
        self.base_url = base_url or BASE_URL
        
        if not self.api_key:
            raise ValueError("请设置 LLM_API_KEY 环境变量或在 .env 文件中配置")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """调用 LLM API 生成回应。"""
        print("正在调用GLM-5大语言模型...")
        try:
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                temperature=0.1
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用 LLM API 时发生错误: {e}")
            raise

def load_boxes_data(json_path: str = None) -> List[Dict]:
    """加载 boxes_2d.json 数据。"""
    if json_path is None:
        json_path = os.path.join(root_dir, 'boxes_2d.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_available_types(boxes: List[Dict]) -> List[str]:
    """获取所有可用的障碍物类型。"""
    types = set()
    for box in boxes:
        if 'type' in box:
            types.add(box['type'])
    return list(types)


def find_box_by_type(boxes: List[Dict], target_type: str) -> Optional[Dict]:
    """根据类型查找障碍物，返回第一个匹配的。"""
    for box in boxes:
        if box.get('type', '') == target_type:
            return box
    return None


def get_first_vertex(box: Dict) -> Optional[Tuple[float, float]]:
    """获取障碍物的第一个角点坐标。"""
    vertices = box.get('vertices', [])
    if vertices:
        return (vertices[0][0], vertices[0][1])
    return None


def get_center_point(box: Dict) -> Optional[Tuple[float, float]]:
    """获取障碍物的中心点坐标。"""
    vertices = box.get('vertices', [])
    if not vertices:
        return None
    x_sum = sum(v[0] for v in vertices)
    z_sum = sum(v[1] for v in vertices)
    n = len(vertices)
    return (x_sum / n, z_sum / n)


def get_safe_navigation_point(
    box: Dict, 
    safety_distance: float = 1.0,
    preferred_direction: str = "auto"
) -> Optional[Tuple[float, float]]:
    """获取物体外部的安全导航点。
    
    计算一个距离物体边界有安全距离的点，确保机器人能够安全到达。
    
    :param box: 障碍物数据，包含 vertices
    :param safety_distance: 安全距离（米），默认 1.0
    :param preferred_direction: 偏好方向 ("auto", "north", "south", "east", "west")
                               auto 会选择物体较短边的方向
    :return: 安全导航点坐标 (x, z)
    """
    vertices = box.get('vertices', [])
    if not vertices or len(vertices) < 3:
        return None
    
    # 计算中心点
    cx = sum(v[0] for v in vertices) / len(vertices)
    cz = sum(v[1] for v in vertices) / len(vertices)
    
    # 计算边界框
    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_z = min(v[1] for v in vertices)
    max_z = max(v[1] for v in vertices)
    
    # 计算物体尺寸
    width_x = max_x - min_x
    width_z = max_z - min_z
    
    # 确定导航点方向
    if preferred_direction == "auto":
        # 选择较短边的方向，这样导航点更接近物体
        if width_x <= width_z:
            # X 方向较短，在 X 方向外侧放置导航点
            preferred_direction = "east"  # 默认向东（+X）
        else:
            # Z 方向较短，在 Z 方向外侧放置导航点
            preferred_direction = "north"  # 默认向北（+Z）
    
    # 计算安全导航点
    if preferred_direction == "north":
        # 物体北侧（+Z 方向）
        nav_x = cx
        nav_z = max_z + safety_distance
    elif preferred_direction == "south":
        # 物体南侧（-Z 方向）
        nav_x = cx
        nav_z = min_z - safety_distance
    elif preferred_direction == "east":
        # 物体东侧（+X 方向）
        nav_x = max_x + safety_distance
        nav_z = cz
    elif preferred_direction == "west":
        # 物体西侧（-X 方向）
        nav_x = min_x - safety_distance
        nav_z = cz
    else:
        # 默认使用中心点外扩
        nav_x = cx
        nav_z = max_z + safety_distance
    
    return (nav_x, nav_z)


def get_all_safe_navigation_points(
    box: Dict, 
    safety_distance: float = 1.0
) -> Dict[str, Tuple[float, float]]:
    """获取物体四个方向的安全导航点。
    
    :param box: 障碍物数据
    :param safety_distance: 安全距离（米）
    :return: 包含四个方向导航点的字典
    """
    vertices = box.get('vertices', [])
    if not vertices or len(vertices) < 3:
        return {}
    
    # 计算中心点
    cx = sum(v[0] for v in vertices) / len(vertices)
    cz = sum(v[1] for v in vertices) / len(vertices)
    
    # 计算边界框
    min_x = min(v[0] for v in vertices)
    max_x = max(v[0] for v in vertices)
    min_z = min(v[1] for v in vertices)
    max_z = max(v[1] for v in vertices)
    
    return {
        "north": (cx, max_z + safety_distance),
        "south": (cx, min_z - safety_distance),
        "east": (max_x + safety_distance, cz),
        "west": (min_x - safety_distance, cz),
    }


def parse_navigation_with_llm(
    user_input: str,
    available_types: List[str],
    llm_client: ZhipuLLMClient = None
) -> Tuple[Optional[str], Optional[str]]:
    """使用智谱 LLM 解析导航指令，提取起点和终点类型。
    
    :param user_input: 用户输入的自然语言，如"从货架到收银台"
    :param available_types: 可用的障碍物类型列表
    :param llm_client: LLM 客户端实例
    :return: (起点类型, 终点类型)
    """
    if llm_client is None:
        llm_client = ZhipuLLMClient()
    
    types_str = "、".join(available_types)
    
    system_prompt = f"""你是一个导航指令解析助手。用户会输入一个导航指令，你需要从中提取起点和终点。

可用的地点类型有：{types_str}

请以 JSON 格式返回结果，包含 "start" 和 "end" 两个字段，值为地点类型名称。
如果用户只指定了终点（如"去收银台"），则 start 为 null。
如果无法识别地点，对应字段为 null。

示例：
输入："从货架到收银台"
输出：{{"start": "货架", "end": "收银台"}}

输入："去收银台"
输出：{{"start": null, "end": "收银台"}}

只返回 JSON，不要有其他文字。"""

    result_text = llm_client.generate(user_input, system_prompt).strip()
    
    # 尝试提取 JSON
    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
    if json_match:
        result_text = json_match.group()
    
    try:
        result = json.loads(result_text)
        return result.get('start'), result.get('end')
    except json.JSONDecodeError:
        print(f"警告：无法解析 LLM 返回的 JSON: {result_text}")
        return None, None


def parse_navigation_simple(user_input: str, available_types: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """简单的规则解析（不依赖 LLM）。
    
    支持格式：
    - "从X到Y"
    - "去X"
    - "到X"
    """
    start_type = None
    end_type = None
    
    # 尝试匹配 "从X到Y" 格式
    match = re.search(r'从(.+?)到(.+)', user_input)
    if match:
        start_candidate = match.group(1).strip()
        end_candidate = match.group(2).strip()
        for t in available_types:
            if t in start_candidate:
                start_type = t
            if t in end_candidate:
                end_type = t
        return start_type, end_type
    
    # 尝试匹配 "去X" 或 "到X" 格式
    match = re.search(r'[去到](.+)', user_input)
    if match:
        end_candidate = match.group(1).strip()
        for t in available_types:
            if t in end_candidate:
                end_type = t
        return None, end_type
    
    # 直接匹配类型名称
    for t in available_types:
        if t in user_input:
            if end_type is None:
                end_type = t
            elif start_type is None:
                start_type = end_type
                end_type = t
    
    return start_type, end_type


def navigate(
    user_input: str,
    use_llm: bool = True,
    boxes_path: str = None,
    llm_client: ZhipuLLMClient = None,
    safety_distance: float = 1.0
) -> Dict:
    """解析导航指令并返回起点和终点的安全导航坐标。
    
    :param user_input: 用户输入的自然语言
    :param use_llm: 是否使用 LLM 解析
    :param boxes_path: boxes_2d.json 文件路径
    :param llm_client: LLM 客户端实例
    :param safety_distance: 安全距离（米），导航点与物体边界的最小距离，默认 1.0
    :return: 包含解析结果的字典
    """
    boxes = load_boxes_data(boxes_path)
    available_types = get_available_types(boxes)
    
    if use_llm:
        start_type, end_type = parse_navigation_with_llm(
            user_input, available_types, llm_client
        )
    else:
        start_type, end_type = parse_navigation_simple(user_input, available_types)
    
    result = {
        "input": user_input,
        "available_types": available_types,
        "safety_distance": safety_distance,
        "parsed": {
            "start_type": start_type,
            "end_type": end_type
        },
        "coordinates": {
            "start": None,
            "end": None
        },
        "all_navigation_points": {
            "start": None,
            "end": None
        }
    }
    
    if start_type:
        start_box = find_box_by_type(boxes, start_type)
        if start_box:
            # 使用安全导航点，而非物体角点
            result["coordinates"]["start"] = get_safe_navigation_point(
                start_box, safety_distance
            )
            # 同时提供四个方向的导航点供选择
            result["all_navigation_points"]["start"] = get_all_safe_navigation_points(
                start_box, safety_distance
            )
    
    if end_type:
        end_box = find_box_by_type(boxes, end_type)
        if end_box:
            # 使用安全导航点，而非物体角点
            result["coordinates"]["end"] = get_safe_navigation_point(
                end_box, safety_distance
            )
            # 同时提供四个方向的导航点供选择
            result["all_navigation_points"]["end"] = get_all_safe_navigation_points(
                end_box, safety_distance
            )
    
    return result


def main():
    parser = argparse.ArgumentParser(description="通过自然语言获取导航坐标（使用智谱 GLM）")
    parser.add_argument("input", type=str, nargs="?", default="从货架到收银台",
                        help="导航指令，如'从货架到收银台'（不传则使用默认）")
    parser.add_argument("--no-llm", action="store_true", help="不使用 LLM，使用简单规则解析")
    parser.add_argument("--boxes", type=str, help="boxes_2d.json 文件路径")
    parser.add_argument("--safety-distance", type=float, default=1.0, 
                        help="安全距离（米），导航点与物体边界的最小距离，默认 1.0")
    args = parser.parse_args()
    
    try:
        # 初始化 LLM 客户端
        llm_client = None
        if not args.no_llm:
            llm_client = ZhipuLLMClient()
        
        result = navigate(
            user_input=args.input,
            use_llm=not args.no_llm,
            boxes_path=args.boxes,
            llm_client=llm_client,
            safety_distance=args.safety_distance
        )
        
        print("\n=== 导航解析结果 ===")
        print(f"输入: {result['input']}")
        print(f"可用地点类型: {result['available_types']}")
        print(f"安全距离: {result['safety_distance']} 米")
        print(f"\n解析结果:")
        print(f"  起点类型: {result['parsed']['start_type']}")
        print(f"  终点类型: {result['parsed']['end_type']}")
        print(f"\n安全导航坐标（距离物体边界 {result['safety_distance']} 米）:")
        print(f"  起点坐标: {result['coordinates']['start']}")
        print(f"  终点坐标: {result['coordinates']['end']}")
        
        # 显示所有方向的导航点
        if result['all_navigation_points']['start']:
            print(f"\n起点可选导航点:")
            for direction, coord in result['all_navigation_points']['start'].items():
                print(f"  {direction}: ({coord[0]:.3f}, {coord[1]:.3f})")
        
        if result['all_navigation_points']['end']:
            print(f"\n终点可选导航点:")
            for direction, coord in result['all_navigation_points']['end'].items():
                print(f"  {direction}: ({coord[0]:.3f}, {coord[1]:.3f})")
        
        print(f"\nJSON 输出:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
