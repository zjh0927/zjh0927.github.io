import base64
import time
import json
from typing import Any, Dict, Optional, Tuple

import cv2
import requests
# 导入ROS 2 Python核心库：创建节点、处理ROS通信
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor #解决HTTP 阻塞
from rclpy.qos import qos_profile_sensor_data
# 导入ROS 2消息类型：传感器图像、字符串、运动指令
from sensor_msgs.msg import Image  # 相机图像消息
from std_msgs.msg import String    # 文本指令消息
from geometry_msgs.msg import Twist  # 机器人运动速度指令
# 导入ROS-OpenCV桥接库：实现ROS Image和OpenCV Mat格式互转
from cv_bridge import CvBridge

from sensor_msgs.msg import LaserScan #激光雷达
import math
import numpy as np
import hashlib

def cv_to_jpeg_b64(cv_img) -> str:
    """
    将OpenCV格式的图像编码为JPEG格式，并转换为Base64字符串（HTTP传输友好）

    Args:
        cv_img: OpenCV格式的图像（numpy.ndarray），BGR通道

    Returns:
        str: 编码后的Base64字符串（UTF-8编码）

    Raises:
        RuntimeError: 图像编码失败时抛出异常
    """
    # cv2.imencode：将OpenCV图像编码为JPEG格式的二进制缓冲区
    # [int(cv2.IMWRITE_JPEG_QUALITY), 80]：设置JPEG质量为80（平衡体积和画质）
    ok, buf = cv2.imencode(".jpg", cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    # 1. buf.tobytes()：将numpy缓冲区转为二进制字节串
    # 2. base64.b64encode()：对二进制数据进行Base64编码
    # 3. decode("utf-8")：将Base64二进制编码转为UTF-8字符串（方便JSON传输）
    return base64.b64encode(buf.tobytes()).decode("utf-8")


class VLABridgeNode(Node):
    """
    VLA桥接节点：实现ROS 2与VLA服务器的通信桥接，核心功能：
    1. 订阅ROS话题：
       - camera_topic (sensor_msgs/Image)：机器人相机原始图像
       - instruction_topic (std_msgs/String)：控制指令（如"前进"、"左转"）
    2. 向VLA服务器发起HTTP POST请求：{vla_url}/act（携带指令+Base64编码的图像）
    3. 发布ROS话题：
       - cmd_vel_topic (geometry_msgs/Twist)：机器人运动速度指令（来自VLA服务器返回）
    """
    def __init__(self):
        super().__init__("vla_bridge_node")  # 初始化ROS 2节点，节点名称为"vla_bridge_node"（标识）

        # ---- 声明ROS 2参数（支持运行时通过--ros-args -p 覆盖，提高灵活性）----
        self.declare_parameter("camera_topic", "/camera/image_raw")  # 相机图像话题名（默认：/camera/image_raw）
        self.declare_parameter("instruction_topic", "/vla/instruction")  # 控制指令话题名（默认：/vla/instruction）
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")  # 机器人速度指令发布话题名（默认：/cmd_vel）
        self.declare_parameter("vla_url", "http://127.0.0.1:8000")  # VLA服务器地址（默认：http://127.0.0.1:8000）若VLA服务器是宿主机必须用宿主机局域网 IP
        self.declare_parameter("rate_hz", 0.2)  # 定时器频率（默认1Hz：每5秒向VLA服务器请求1次）
        self.declare_parameter("http_timeout_s", 25.0)  #8.0 HTTP请求超时时间（默认8秒：防止请求卡住）
        self.declare_parameter("lin_scale", 1.0)
        self.declare_parameter("ang_scale", 1.0)
        self.declare_parameter("max_lin", 0.26)   # TurtleBot3 常见安全量级
        self.declare_parameter("max_ang", 1.82)   # 教学/实践常用安全量级
        self.declare_parameter("accel_lin", 0.5)  # m/s^2，0 表示不做加速度限制
        self.declare_parameter("accel_ang", 2.0)  # rad/s^2
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("front_angle", 0.0)  # rad，默认认为 scan 的 0 就是正前方
        self.declare_parameter("obstacle_dist", 0.40)   # 前方小于 0.4m 认为危险
        self.declare_parameter("control_hz", 10.0)      # 小脑控制频率
        self.declare_parameter("remote_fresh_s", 6.0)   # 0 表示自动推导
        self.declare_parameter("stop_when_no_instruction", True)  # 无指令时是否停止机器人（默认True：安全保护）
        self.declare_parameter("send_sensor_context", True)  # 是否默认发送传感器摘要给VLM

        # 读取参数值并赋值给实例变量（方便后续调用）
        self.camera_topic = self.get_parameter("camera_topic").value
        self.instruction_topic = self.get_parameter("instruction_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.vla_url = self.get_parameter("vla_url").value.rstrip("/")
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.http_timeout_s = float(self.get_parameter("http_timeout_s").value)
        self.lin_scale = float(self.get_parameter("lin_scale").value)
        self.ang_scale = float(self.get_parameter("ang_scale").value)
        #突破极限要知道Gazebo 差速驱动插件（diff drive plugin），很多项目是把插件放在单独的 *.gazebo.xacro 或 Gazebo 的 model.sdf 里
        self.max_lin = float(self.get_parameter("max_lin").value)#不突破仿真控制器的最大速度上限
        self.max_ang = float(self.get_parameter("max_ang").value)#不突破仿真控制器的最大角速度上限
        self.accel_lin = float(self.get_parameter("accel_lin").value)
        self.accel_ang = float(self.get_parameter("accel_ang").value)
        self.front_angle = float(self.get_parameter("front_angle").value)
        self.scan_topic = self.get_parameter("scan_topic").value
        self.obstacle_dist = float(self.get_parameter("obstacle_dist").value)
        self.control_hz = float(self.get_parameter("control_hz").value)
        self.send_sensor_context = bool(self.get_parameter("send_sensor_context").value)

        remote_fresh_s = float(self.get_parameter("remote_fresh_s").value)

        # ✅ 自动推导：给网络/推理抖动留余量，但又不会太久
        if remote_fresh_s <= 0.0:
            # 只跟 tick 周期相关：信任最近 1~2 个周期即可，不要被 timeout 拉长
            auto = max(1.6 / max(self.rate_hz, 0.1), 0.5)  # 约等于 1~2 个周期
            auto = min(auto, 5.0)                           # 上限 5s，避免旧动作拖太久
            self.remote_fresh_s = auto
        else:
            self.remote_fresh_s = remote_fresh_s

        self._inflight = False
        self._next_try_t = 0.0


        self._last_cmd_lin = 0.0
        self._last_cmd_ang = 0.0
        self._last_tick_t = time.time()
        self.latest_scan: Optional[LaserScan] = None
        self.latest_scan_t: float = 0.0
        self.remote_action: Optional[tuple[float, float, float]] = None  # (lin, ang, t)
        self.stop_when_no_instruction = bool(self.get_parameter("stop_when_no_instruction").value)

        self.bridge = CvBridge()  # 初始化CvBridge：用于ROS Image <-> OpenCV Mat格式转换
        self.latest_img: Optional[Image] = None  # 存储最新的相机图像（初始为None：表示还未接收到图像）
        self.latest_instruction: str = ""  # 存储最新的控制指令（初始为空字符串）

        # ---- 创建ROS 2订阅器 ----
        # 订阅相机图像话题，回调函数on_image，队列大小10（缓存最多10条消息）
        self.sub_img = self.create_subscription(Image, self.camera_topic, self.on_image, qos_profile_sensor_data)#10

        # 订阅控制指令话题，回调函数on_instruction，队列大小10
        self.sub_inst = self.create_subscription(String, self.instruction_topic, self.on_instruction, 10)

        self.sub_scan = self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)#10

        # ---- 创建ROS 2发布器 ----
        self.pub_cmd = self.create_publisher(Twist, self.cmd_vel_topic, 10)  # 发布机器人速度指令，队列大小10

        period = 1.0 / max(self.rate_hz, 0.1)  # 计算定时器周期（秒），max避免rate_hz为0导致除零错误
        self.timer = self.create_timer(period, self.tick)  # 创建定时器：每隔period秒执行一次tick回调函数（核心业务逻辑）

        control_period = 1.0 / max(self.control_hz, 1.0)
        self.control_timer = self.create_timer(control_period, self.control_tick)

        # 打印节点初始化信息（方便调试，确认参数是否正确加载）
        self.get_logger().info(f"camera_topic={self.camera_topic}")
        self.get_logger().info(f"instruction_topic={self.instruction_topic}")
        self.get_logger().info(f"cmd_vel_topic={self.cmd_vel_topic}")
        self.get_logger().info(f"vla_url={self.vla_url} (POST {self.vla_url}/act)")

    def on_image(self, msg: Image):
        """
        相机图像话题回调函数：接收到新图像时，更新最新图像缓存

        Args:
            msg: ROS 2 sensor_msgs/Image类型的消息（相机原始图像）
        """
        # 不改 launch：用 Unix 时间覆盖 ROS 相对时间
        now = time.time()
        msg.header.stamp.sec = int(now)
        msg.header.stamp.nanosec = int((now - int(now)) * 1e9)
        self.latest_img = msg

    def on_instruction(self, msg: String):
        """
        控制指令话题回调函数：接收到新指令时，更新最新指令缓存

        Args:
            msg: ROS 2 std_msgs/String类型的消息（控制指令文本）
        """
        self.latest_instruction = (msg.data or "").strip()  # strip()：去除指令前后的空格/换行符，避免无效指令
        self.get_logger().info(f"Instruction updated: {self.latest_instruction!r}")  # 打印指令更新日志（方便调试，确认指令是否正确接收）

        # 指令一变：立即让旧的远端动作失效，避免继续跑旧动作
        self.remote_action = None  # (lin, ang, t)

    def on_scan(self, msg: LaserScan):
        self.latest_scan = msg
        self.latest_scan_t = time.time()

    def publish_stop(self):
        """发布停止指令：让机器人线性速度和角速度都为0（紧急停止/无指令时使用）"""
        tw = Twist()  # 创建空的Twist消息
        tw.linear.x = 0.0  # 线性速度（前进/后退）为0
        tw.angular.z = 0.0  # 角速度（左转/右转）为0
        self.pub_cmd.publish(tw)  # 发布停止指令
        self.get_logger().debug("🛑 发布停止指令")

    def tick(self):
        """
        大脑请求定时器（低频）：只负责向VLA服务器请求动作，并缓存结果
        ✅ 不在这里发布 /cmd_vel，避免和 control_tick(小脑) 打架
        """
        # 1) 无控制指令时：不请求大脑（小脑会负责 stop）
        if not self.latest_instruction:
            return

        now = time.time()
        # 退避：上一轮失败后，短时间内不重试，避免失败风暴
        if now < self._next_try_t:
            return

        # 2) 无相机图像时：跳过请求（小脑仍可用雷达/灰度 fallback）
        if self.latest_img is None:
            self.get_logger().warn("⚠️ 未接收到相机图像，跳过本次大脑请求")
            return

        # 单飞：1.上一次请求没完成就不再发，避免堆积/并发/乱序 2.如果图像过旧（或无更新），跳过请求
        img_time = self.latest_img.header.stamp.sec + self.latest_img.header.stamp.nanosec * 1e-9#用计算好的img_time（完整时间戳）代替stamp.sec
        self.get_logger().info(f"time.time() - img_time({img_time})={time.time() - img_time}") 
        if self._inflight or time.time() - img_time > 8.0:#1.0 判断旧图的间隔的时间根据模型推理的速度改动，越慢越大
            return
        self._inflight = True

        self.get_logger().debug(f"Image stamp: {self.latest_img.header.stamp.sec}.{self.latest_img.header.stamp.nanosec}")

        # 3) ROS Image → OpenCV
        try:
            cv_img = self.bridge.imgmsg_to_cv2(self.latest_img, desired_encoding="bgr8")
            self.get_logger().debug(f"img shape={cv_img.shape}, mean={cv2.mean(cv_img)}")
        except Exception as e:
            self.get_logger().warn(f"❌ 图像格式转换失败: {e}")
            # ========== 新增：释放_inflight后再return ==========
            self._inflight = False
            return

        # 4) OpenCV图像 → Base64（HTTP传输）
        try:
            image_b64 = cv_to_jpeg_b64(cv_img)
            img_md5 = hashlib.md5(base64.b64decode(image_b64.encode("utf-8"))).hexdigest()
            self.get_logger().info(f"[IMG] stamp={self.latest_img.header.stamp.sec}.{self.latest_img.header.stamp.nanosec} md5={img_md5}")
        except Exception as e:
            self.get_logger().warn(f"❌ 图像Base64编码失败: {e}")
            # ========== 新增：释放_inflight后再return ==========
            self._inflight = False
            return
        # ========== 修改2：删除Base64编码后的finally（提前释放_inflight的问题） ==========

        # 5) 构造请求体
        payload = {
            "instruction": self.latest_instruction,
            "image_b64": image_b64,
            "timestamp": time.time(),
        }
        if self.send_sensor_context:
            # 默认携带雷达摘要，减少传输量且保留上下文安全性
            payload["sensor_context"] = {
                "source": "laser_scan",
                "scan": self._scan_summary(self.latest_scan),
            }

        # 6) 请求VLA服务器（大脑）
        try:
            resp = requests.post(
                f"{self.vla_url}/act",
                json=payload,
                timeout=self.http_timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            # 大脑慢/失败没关系：小脑会继续用本地避障输出
            self.get_logger().warn(f"❌ 大脑请求失败(将继续小脑避障): {e}")
            self._next_try_t = time.time() + 1.0   # ✅ 失败退避 1s（可调）
            self._inflight = False  # ========== 新增：释放_inflight ==========
            return #finally先执行
        finally:
            self._inflight = False

        # 7) 解析响应并缓存（注意：这里不做 scale/clamp/ramp）
        action = data.get("action", {})
        lin = float(action.get("linear_x", 0.0))
        ang = float(action.get("angular_z", 0.0))

        # 缓存远端动作与时间戳（供 control_tick 使用）
        t = time.time()
        self.remote_action = (lin, ang, t)
        self.get_logger().info(f"✅ 接收大脑指令：linear_x={self.remote_action[0]:.3f}, angular_z={self.remote_action[1]:.3f} (有效期至：{t + self.remote_fresh_s:.2f})")

    def _sector_min(self, scan: LaserScan, center_angle: float, half_width: float) -> float:
        # 返回某个扇区内的最小距离（无有效值则 inf）
        if scan is None or not scan.ranges:
            return float("inf")
        inc = scan.angle_increment
        if inc <= 0:
            return float("inf")

        # TB3 仿真常见 angle_min=0, angle_max=2pi；也兼容 -pi..pi
        vals = []
        rmin = scan.range_min if scan.range_min > 0 else 0.0
        rmax = scan.range_max if scan.range_max > 0 else float("inf")
        a = scan.angle_min
        for r in scan.ranges:
            if np.isfinite(r) and (r > rmin) and (r < rmax):
                # 角度差归一化到 [-pi, pi]
                d = (a - center_angle + math.pi) % (2*math.pi) - math.pi
                if abs(d) <= half_width:
                    vals.append(r)
            a += inc
        return float(min(vals)) if vals else float("inf")

    def _scan_summary(self, scan: Optional[LaserScan]) -> Optional[Dict[str, Any]]:
        # 把雷达扇区关键距离压缩成少量指标，避免把几百个 ranges 全发给VLM
        if scan is None:
            return None

        valid_ranges = [r for r in scan.ranges if np.isfinite(r) and r > 0]
        if not valid_ranges:
            return {
                "ok": False,
                "reason": "no finite ranges",
                "angle_min": float(scan.angle_min),
                "angle_max": float(scan.angle_max),
                "angle_increment": float(scan.angle_increment),
                "range_min": float(scan.range_min),
                "range_max": float(scan.range_max),
            }

        front = self._sector_min(scan, center_angle=self.front_angle, half_width=math.radians(15))
        left = self._sector_min(
            scan,
            center_angle=self.front_angle + math.pi / 2.0,
            half_width=math.radians(20),
        )
        right = self._sector_min(
            scan,
            center_angle=self.front_angle - math.pi / 2.0,
            half_width=math.radians(20),
        )

        sorted_ranges = sorted(valid_ranges)
        scan_seq = [float(v) for v in scan.ranges]
        return {
            "ok": True,
            "stamp_sec": float(scan.header.stamp.sec),
            "stamp_nsec": float(scan.header.stamp.nanosec),
            "angle_min": float(scan.angle_min),
            "angle_max": float(scan.angle_max),
            "angle_increment": float(scan.angle_increment),
            "range_min": float(scan.range_min),
            "range_max": float(scan.range_max),
            "ranges_valid_ratio": float(len(valid_ranges)) / max(len(scan_seq), 1),
            "global_min_m": float(sorted_ranges[0]),
            "global_med_m": float(sorted_ranges[len(sorted_ranges) // 2]) if sorted_ranges else float("nan"),
            "front_min_m": float(front),
            "left_min_m": float(left),
            "right_min_m": float(right),
            "obstacle_front": bool(front < self.obstacle_dist),
            "scan_age_s": float(time.time() - self.latest_scan_t) if self.latest_scan_t > 0 else None,
        }

    def _lidar_avoid_needed(self) -> bool:
        """
        判断是否需要雷达硬避障（永远优先）：
        - 取正前方一个扇区的最小距离
        - 小于 obstacle_dist 则认为必须避障
        """
        sc = self.latest_scan
        if sc is None:
            return False

        front = self._sector_min(sc, center_angle=self.front_angle, half_width=math.radians(15))
        return front < self.obstacle_dist

    def fallback_action(self) -> tuple[float, float, bool]:
        """
        小脑本地策略（高频、实时）：
        1) 雷达硬避障：只要前方太近，立刻转向（hard_avoid=True）
        2) 否则可选用灰度规则微调
        返回：(linear_x, angular_z, hard_avoid)
        """
        # 默认慢速前进
        lin, ang = 0.15, 0.0
        hard_avoid = False

        sc = self.latest_scan
        if sc is not None:
            front_center = self.front_angle
            left_center  = self.front_angle + math.pi / 2.0
            right_center = self.front_angle - math.pi / 2.0

            # 前方 ±15°
            front = self._sector_min(sc, center_angle=front_center, half_width=math.radians(15))
            # 左右扇区：用于决定往哪边转更安全
            left  = self._sector_min(sc, center_angle=left_center,  half_width=math.radians(20))
            right = self._sector_min(sc, center_angle=right_center, half_width=math.radians(20))

            if front < self.obstacle_dist:
                # ✅ 雷达触发：硬避障永远优先
                hard_avoid = True
                lin = 0.0
                ang = +0.9 if left > right else -0.9
                return lin, ang, hard_avoid

        # 雷达未触发：可选灰度规则（例如画面过暗就转一下）
        if self.latest_img is not None:
            try:
                cv_img = self.bridge.imgmsg_to_cv2(self.latest_img, desired_encoding="bgr8")
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                center = gray[
                    gray.shape[0] // 3 : 2 * gray.shape[0] // 3,
                    gray.shape[1] // 3 : 2 * gray.shape[1] // 3,
                ]
                if float(np.mean(center)) < 60:
                    return 0.0, 0.8, False
            except Exception:
                pass

        return lin, ang, hard_avoid

    def control_tick(self):
        """
        小脑控制定时器（高频）：始终发布 /cmd_vel
        优先级（从高到低）：
        1) 无指令：stop（安全）
        2) 雷达硬避障：永远优先（不允许大脑覆盖）
        3) 大脑动作（若足够新鲜）
        4) 本地fallback（灰度/慢速前进）
        并在最终发布前统一做：scale/clamp/ramp
        """
        # 0) 无指令：立即停
        if not self.latest_instruction:
            if self.stop_when_no_instruction:
                self.publish_stop()
            return

        now = time.time()

        # 1) 先算本地 fallback，并知道是否触发"雷达硬避障"
        fb_lin, fb_ang, hard_avoid = self.fallback_action()

        # 2) 决定是否使用大脑动作
        lin, ang = fb_lin, fb_ang
        ra = self.remote_action  # 只读一次，避免中途被替换
        remote_fresh = (ra is not None) and ((now - ra[2]) < self.remote_fresh_s)

        if hard_avoid:
            # ✅ 雷达硬避障永远优先：大脑绝不能覆盖
            lin, ang = fb_lin, fb_ang
            self.get_logger().info(f"🔧 小脑使用雷达硬避障")
        else:
            # 雷达未触发：如果大脑新鲜就用大脑，否则用fallback
            if remote_fresh:
                lin, ang = float(ra[0]), float(ra[1])
                self.get_logger().info(f"🔧 小脑使用大脑指令：lin={lin:.3f}, ang={ang:.3f}")
            else:
                self.get_logger().info(f"🔧 小脑使用fallback指令：lin={lin:.3f}, ang={ang:.3f}")

        # 3) 统一做 scale + clamp（最终发给底盘的速度）
        lin *= self.lin_scale
        ang *= self.ang_scale
        lin = max(-self.max_lin, min(self.max_lin, lin))
        ang = max(-self.max_ang, min(self.max_ang, ang))

        # 4) ramp：用 control_tick 的 dt 进行加速度限制（平滑）
        dt = max(1e-3, now - self._last_tick_t)
        self._last_tick_t = now

        if self.accel_lin > 0:
            step = self.accel_lin * dt
            lin = max(self._last_cmd_lin - step, min(self._last_cmd_lin + step, lin))

        if self.accel_ang > 0:
            step = self.accel_ang * dt
            ang = max(self._last_cmd_ang - step, min(self._last_cmd_ang + step, ang))

        self._last_cmd_lin = lin
        self._last_cmd_ang = ang

        # 5) 发布速度
        tw = Twist()
        tw.linear.x = lin
        tw.angular.z = ang
        self.pub_cmd.publish(tw)    

def main():
    """ROS 2节点主函数：标准启动流程"""
    rclpy.init()  # 初始化ROS 2上下文
    node = VLABridgeNode()  # 创建VLA桥接节点实例
    executor = MultiThreadedExecutor(num_threads=2)  # 最小：2线程足够
    executor.add_node(node)
    try:
        #rclpy.spin(node)  # 自旋节点：持续处理回调函数（订阅/定时器），阻塞直到节点关闭
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()  # 确保节点正常销毁（释放资源）
        rclpy.shutdown()  # 关闭ROS 2上下文

if __name__ == "__main__":
    main()
