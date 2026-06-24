ROS2+Gazebo+VLM：纯仿真环境下的具身智能闭环实现
==============================================

大脑-小脑分层控制架构

## 架构

```
指令发布 (instruction_publisher.py)
    ↓ /vla/instruction
VLA Agent (vla_agent_node2.py)
    ↓ HTTP POST /act
VLM Server (vla_server2.py)
    ↓ 返回动作 {linear_x, angular_z}
Agent 控制层
    ↓ /cmd_vel
Gazebo 仿真机器人
```

## 频率分离

| 组件 | 频率 | 周期 |
|------|------|------|
| 大脑 (VLM 推理) | 0.2 Hz | 5 s |
| 小脑 (运动控制) | 10 Hz | 100 ms |

## 安全机制

1. **雷达硬避障** (最高优先级) — 前方 0.4m 以内立即转向
2. **大脑指令** (次优先级) — 8 秒内有效
3. **本地兜底** (最低优先级) — 0.15 m/s 慢速前进

## 运行

```bash
# 1. 启动 VLM 推理服务器 (宿主机)
python vla_server2.py --port 5000 --model Qwen2.5-VL-7B

# 2. 启动 Gazebo 仿真 + ROS2 节点 (Ubuntu 虚拟机)
ros2 run vla_agent vla_agent_node2 --ros-args -p server_url:=http://192.168.1.100:5000

# 3. 发布指令
python instruction_publisher.py "走到厨房"
```

## 参考

- 原文: https://mp.weixin.qq.com/s/VdD5H65Bug4zQ5Qq-cZZNw
- CSDN: https://blog.csdn.net/weixin_55221858/article/details/156659624
