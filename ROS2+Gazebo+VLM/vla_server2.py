import hashlib
import os
import re
import json
import time
import base64
import tempfile
import asyncio
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

id=1

# =============== 可选：本地 Transformers Qwen3-VL ===============
LOCAL_VLM_AVAILABLE = False
try:
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    LOCAL_VLM_AVAILABLE = True
except Exception:
    LOCAL_VLM_AVAILABLE = False


# -------------------- 配置 --------------------
BACKEND = os.getenv("VLA_BACKEND", "cloud").strip().lower()
# BACKEND: "rule" | "local-model" | "ollama" | "cloud"

# local-model 配置
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", r"../model/Qwen3-VL-2B-Instruct")
LOCAL_MAX_NEW_TOKENS = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "64"))#128
LOCAL_TOP_P = float(os.getenv("LOCAL_TOP_P", "0.9"))#0.8 # 仅选择累计概率80%的token
LOCAL_TEMPERATURE = float(os.getenv("LOCAL_TEMPERATURE", "0.7"))#0.2 极低温度，压缩概率分布

# ollama 配置
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl")  # 需要是支持图像输入的模型
OLLAMA_TIMEOUT_S = float(os.getenv("OLLAMA_TIMEOUT_S", "20"))

# cloud 配置（兼容 OpenAI 风格的 Chat Completions 接口）
CLOUD_BASE_URL = os.getenv(
    "CLOUD_BASE_URL",
    os.getenv("MIMO_BASE_URL", ""),
).rstrip("/")
CLOUD_API_KEY = os.getenv(
    "CLOUD_API_KEY",
    os.getenv("MIMO_API_KEY", ""),
)
CLOUD_MODEL = os.getenv(
    "CLOUD_MODEL", os.getenv("MIMO_MODEL", "mimo-v2.5")
)
CLOUD_TIMEOUT_S = float(os.getenv("CLOUD_TIMEOUT_S", "30"))
CLOUD_MAX_TOKENS = int(os.getenv("CLOUD_MAX_TOKENS", "1024"))
CLOUD_TEMPERATURE = float(os.getenv("CLOUD_TEMPERATURE", "0.2"))

# 速度/安全限制：Twist 范围（可按需要调）
MAX_LIN = float(os.getenv("VLA_MAX_LIN", "5.5"))
MAX_ANG = float(os.getenv("VLA_MAX_ANG", "2.5"))


# -------------------- FastAPI --------------------
app = FastAPI(title="VLA Server (rule/local-model/ollama/cloud)")


class ActRequest(BaseModel):
    instruction: str
    image_b64: str
    timestamp: Optional[float] = None
    # 可选：由ROS侧上报的传感器摘要（如激光雷达压缩信息），用于增强决策输入
    sensor_context: Optional[Dict[str, Any]] = None


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def decode_image_b64_to_bgr(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64.encode("utf-8"))
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed")
    return img


# -------------------- 1) rule 后端 --------------------
def rule_vla(instruction: str, image_bgr: np.ndarray) -> Dict[str, Any]:
    t = (instruction or "").lower()

    if "stop" in t or "halt" in t:
        return {"linear_x": 0.0, "angular_z": 0.0}
    if "left" in t:
        return {"linear_x": 0.0, "angular_z": +0.8}
    if "right" in t:
        return {"linear_x": 0.0, "angular_z": -0.8}
    if "back" in t:
        return {"linear_x": -0.15, "angular_z": 0.0}

    # 简单"明暗避障"
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    center = gray[
        gray.shape[0] // 3 : 2 * gray.shape[0] // 3,
        gray.shape[1] // 3 : 2 * gray.shape[1] // 3,
    ]
    mean_intensity = float(np.mean(center))
    if mean_intensity < 60:
        return {"linear_x": 0.0, "angular_z": 0.9}
    return {"linear_x": 0.20, "angular_z": 0.0}


# -------------------- VLM 输出 -> action 解析 --------------------
_JSON_RE = re.compile(r"\{[\s\S]*\}")

def parse_action_from_text(text: str) -> Optional[Dict[str, float]]:
    """
    期望模型输出 JSON:
      {"linear_x": 0.1, "angular_z": -0.2}
    容错：从文本里抓第一个 {...} 解析。失败返回 None。
    """
    if not text:
        return None

    m = _JSON_RE.search(text)
    if not m:
        return None

    try:
        obj = json.loads(m.group(0))
        lin = float(obj.get("linear_x", 0.0))
        ang = float(obj.get("angular_z", 0.0))

        # 限幅，避免模型发疯
        lin = clamp(lin, -MAX_LIN, MAX_LIN)
        ang = clamp(ang, -MAX_ANG, MAX_ANG)
        return {"linear_x": lin, "angular_z": ang}
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def build_action_prompt(instruction: str, sensor_context: Optional[Dict[str, Any]] = None) -> str:
    # 强制 JSON-only 输出：这是把"VLM 文本"变成"动作"的关键
    # return (
    #     "You are a robot control policy. "
    #     "Given the user's instruction and the image, output ONLY a JSON object.\n"
    #     "JSON schema:\n"
    #     '{ "linear_x": number, "angular_z": number }\n'
    #     f"Constraints: linear_x in [-{MAX_LIN}, {MAX_LIN}], angular_z in [-{MAX_ANG}, {MAX_ANG}].\n"
    #     "No extra text.\n"
    #     f"Instruction: {instruction}\n"
    # )
    # 通用机器人视觉控制，核心转弯/退后避障
    return (
        "You are a robot motion control policy for a differential-drive robot.\n"
        "Given a front-view camera image and the user instruction, output ONLY ONE JSON object:\n"
        '{"linear_x":<number>,"angular_z":<number>}\n\n'

        "### PRIMARY GOAL (highest priority)\n"
        "- Treat a WHITE CYLINDER (bright white circular/oval pillar) as the TARGET.\n"
        "- If the target is visible: approach it safely.\n"
        "- When the target is very close (it touches the bottom edge OR occupies a large area in the lower center), STOP.\n\n"

        "### Target detection rules (use IMAGE cues)\n"
        "- Target is a bright white cylinder: a white circular/oval blob with smooth boundary.\n"
        "- Estimate horizontal error by target position:\n"
        "  * target left of center -> turn left (angular_z > 0)\n"
        "  * target right of center -> turn right (angular_z < 0)\n"
        "  * target near center -> go straight (angular_z ~= 0)\n"
        "- Estimate closeness by size/vertical position:\n"
        "  * small and high in image -> far\n"
        "  * large and low in image -> close\n"
        "  * touches bottom edge / very large in lower center -> reached\n\n"

        "### Target pursuit control (MUST follow if target visible)\n"
        f"- Output ranges: linear_x in [-{MAX_LIN}, {MAX_LIN}], angular_z in [-{MAX_ANG}, {MAX_ANG}]\n"
        "- If target reached (very close): linear_x=0.0 and angular_z=0.0\n"
        "- Else if target visible but not centered:\n"
        "  * far: linear_x in [2.0,3.0], angular_z in [+0.8,+1.6] (if target left) or [-1.6,-0.8] (if target right)\n"
        "  * close: linear_x in [0.8,1.6], angular_z in [+0.6,+1.2] or [-1.2,-0.6]\n"
        "- Else if target visible and centered:\n"
        "  * far: linear_x in [3.0,4.5], angular_z=0.0\n"
        "  * close: linear_x in [1.0,2.0], angular_z=0.0\n\n"

        "### SAFETY OVERRIDE\n"
        "- If a wall/obstacle touches the bottom edge in the center and blocks motion, avoid collision even if target visible.\n"
        "- If collision imminent: set linear_x=0.0 and turn in place with angular_z in [+1.2,+1.8] or [-1.8,-1.2].\n\n"

        "### Fallback navigation (use ONLY if target NOT visible)\n"
        "### Distance & Geometry Rules (CRITICAL)\n"
        "- Objects that appear LOWER in the image are CLOSER to the robot.\n"
        "- Larger image area (higher pixel coverage) usually means the object is CLOSER.\n"
        "- If walls/objects touch the bottom edge, treat as CENTER_CLOSE.\n"
        "- If large regions enter from the bottom corners, treat as CENTER_CLOSE (near a corner).\n\n"

        "### Step 1: Decide the scene category from the IMAGE (must choose ONE)\n"
        "A) CENTER_CLOSE: obstacle/wall touches the bottom edge or occupies the lower half\n"
        "B) CENTER_FAR: obstacle exists in the center but still has distance (can steer around)\n"
        "C) LEFT_MORE_OPEN: left side is more open than right side\n"
        "D) RIGHT_MORE_OPEN: right side is more open than left side\n"
        "E) CLEAR: center and both sides look open (safe to go forward)\n"
        "F) UNCERTAIN: cannot judge (blur/dark/ambiguous)\n\n"

        "### Step 2: Map category to motion (must follow exactly)\n"
        "- CENTER_CLOSE: choose ONE of:\n"
        "  * turn in place: linear_x=0.0 and angular_z in [+1.2,+1.8] or [-1.2,-1.8]\n"
        "  * or back up: linear_x in [-0.6,-1.2] and angular_z in [-0.6,+0.6]\n"
        "- CENTER_FAR: DO NOT STOP. steer around while moving forward:\n"
        "  * linear_x in [1.6,2.8], angular_z in [+0.6,+1.2] OR [-1.2,-0.6]\n"
        "- LEFT_MORE_OPEN: linear_x in [2.0,3.2], angular_z in [+0.6,+1.4]\n"
        "- RIGHT_MORE_OPEN: linear_x in [2.0,3.2], angular_z in [-1.4,-0.6]\n"
        "- CLEAR: linear_x in [3.0,4.5], angular_z=0.0\n"
        "- UNCERTAIN: move slowly and search (do not output 0,0):\n"
        "  * linear_x in [0.8,1.6], angular_z in [-0.6,+0.6]\n\n"

        "### Instruction following (apply after safety)\n"
        "- If instruction says STOP/HALT, output linear_x=0.0 and angular_z=0.0.\n\n"

        "### Strict output requirement\n"
        "Output ONLY a single valid JSON object, no extra text.\n\n"
        f"### User instruction\n{instruction}\n"
    )

    if sensor_context:
        sensor_block = json.dumps(sensor_context, ensure_ascii=False, separators=(",", ":"))
        # 可选上下文以 JSON 字符串形式附加到提示词，便于模型做"视觉+传感器"联合判断
        return (
            prompt
            + "\n### SENSOR CONTEXT (optional)\n"
            + "Use instruction, image, and this sensor context jointly for action planning.\n"
            + f"{sensor_block}\n"
        )

    return prompt

# -------------------- 2) local-model 后端 --------------------
def get_device() -> "torch.device":
    # 宿主机 macOS：优先 MPS；Linux VM 通常没有 MPS
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def local_model_infer(prompt: str, image_bytes: bytes) -> str:
    """
    用 temp 文件喂给 Qwen3-VL 的 processor（原代码使用 image_path）
    """
    suffix = ".jpg"
    tmp_dir = Path(tempfile.gettempdir()) / "vla_upload"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"vla_{int(time.time()*1000)}{suffix}"
    tmp_path.write_bytes(image_bytes)

    try:
        content = [{"type": "image", "image": str(tmp_path)}, {"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]

        inputs = app.state.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = inputs.to(app.state.device)

        with torch.no_grad():
            generated_ids = app.state.model.generate(
                **inputs,
                max_new_tokens=LOCAL_MAX_NEW_TOKENS,
                do_sample=(LOCAL_TEMPERATURE > 0),
                top_p=LOCAL_TOP_P,
                temperature=LOCAL_TEMPERATURE,
            )

        input_ids = inputs["input_ids"]
        new_token_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
        out_text = app.state.processor.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return out_text
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# -------------------- 3) ollama 后端 --------------------
def ollama_infer(prompt: str, image_b64: str) -> str:
    """
    使用 /api/generate：支持 images: [base64]，并支持 format=json 结构化输出
    """
    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "images": [image_b64],   # base64 数组
        "stream": False,
        "format": "json",        # 让输出是合法 JSON
    }
    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    # /api/generate 在非流式时一般返回 {"response": "...", "done": true, ...}
    return data.get("response", "")


# -------------------- 4) cloud 后端 --------------------
def cloud_infer(prompt: str, image_b64: str) -> str:
    # 使用 OpenAI 风格的 chat/completions 接口发起图文推理
    url = f"{CLOUD_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {CLOUD_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CLOUD_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": CLOUD_MAX_TOKENS,
        "temperature": CLOUD_TEMPERATURE,
        "stream": False,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=CLOUD_TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    # 常见返回是 choices[0].message.content
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")


# -------------------- FastAPI 生命周期：按后端加载模型 --------------------
@app.on_event("startup")
def _startup():
    app.state.backend = BACKEND
    app.state.lock = asyncio.Lock()

    if BACKEND == "local-model":
        if not LOCAL_VLM_AVAILABLE:
            raise RuntimeError(
                "BACKEND=local-model but transformers/torch/Qwen3VL imports failed. "
                "Install deps first."
            )
        app.state.device = get_device()
        print(f"[INFO] local-model device = {app.state.device}")

        # Qwen3-VL 的 transformers 版本经常需要较新/源码版（官方卡片有提示）
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            LOCAL_MODEL_DIR,
            torch_dtype=(torch.float16 if app.state.device.type in {"cuda", "mps"} else torch.float32),
            device_map="auto" if app.state.device.type != "cpu" else None,
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(LOCAL_MODEL_DIR)

        app.state.model = model
        app.state.processor = processor
        print(f"[INFO] local-model loaded from: {LOCAL_MODEL_DIR}")

    elif BACKEND == "ollama":
        print(f"[INFO] ollama backend: {OLLAMA_URL}, model={OLLAMA_MODEL}")
    elif BACKEND == "cloud":
        if not CLOUD_API_KEY:
            raise RuntimeError("BACKEND=cloud but CLOUD_API_KEY is empty.")
        print(f"[INFO] cloud backend: model={CLOUD_MODEL}")#{CLOUD_BASE_URL},
    else:
        print("[INFO] rule backend enabled")


@app.get("/health")
def health():
    info = {"ok": True, "backend": BACKEND}
    if BACKEND == "local-model" and hasattr(app.state, "device"):
        info["device"] = str(app.state.device)
        info["model_dir"] = LOCAL_MODEL_DIR
    if BACKEND == "ollama":
        info["ollama_url"] = OLLAMA_URL
        info["ollama_model"] = OLLAMA_MODEL
    if BACKEND == "cloud":
        info["cloud_base_url"] = CLOUD_BASE_URL
        info["cloud_model"] = CLOUD_MODEL
    return info

@app.post("/act")
async def act(req: ActRequest):
    t0 = time.time()          # ← 起始计时

    # 1) 解码图像（BGR）用于 rule 后端；同时保留 bytes 给 VLM
    try:
        img_bgr = decode_image_b64_to_bgr(req.image_b64)
        img_bytes = base64.b64decode(req.image_b64.encode("utf-8"))
        # gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # mean = float(np.mean(gray))
        # h, w = gray.shape[:2]
        # print(f"[DEBUG] image={w}x{h}, gray_mean={mean:.1f}")

        # # 每 5 秒存一张，方便打开确认
        # if int(time.time()) % 5 == 0:
        #     global id
        #     cv2.imwrite("tmp-data/vla_{id}.jpg".format(id=id), img_bgr)
        #     print('[DEBUG]保存图像')
        #     id+=1
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=f"Bad image_b64: {e}")

    instruction = (req.instruction or "").strip()
    if not instruction:
        raise HTTPException(status_code=400, detail="instruction is empty")

    # 2) 后端分发
    if BACKEND == "rule":
        action = rule_vla(instruction, img_bgr)
        return {"action": action, "model": "rule_vla_stub"}

    prompt = build_action_prompt(instruction, req.sensor_context)

    try:
        async with app.state.lock:
            if BACKEND == "local-model":
                global id
                cv2.imwrite("tmp-data/vla_{id}.jpg".format(id=id), img_bgr)
                out_text = local_model_infer(prompt, img_bytes)
                md5 = hashlib.md5(img_bytes).hexdigest()
                print(f'DEBUG] id={id} md5={md5} 本地模型输出：',out_text)
                id+=1
                action = parse_action_from_text(out_text)
                print(f'本地模型的动作解析结果：{action} 用时{time.time() - t0}')
                return {"action": action, "model": "local-model", "raw": out_text}

            if BACKEND == "ollama":
                out_text = ollama_infer(prompt, req.image_b64)
                action = parse_action_from_text(out_text)
                return {"action": action, "model": f"ollama:{OLLAMA_MODEL}", "raw": out_text}
            if BACKEND == "cloud":
                out_text = cloud_infer(prompt, req.image_b64)
                action = parse_action_from_text(out_text) or {"linear_x": 0.0, "angular_z": 0.0}
                elapsed_ms = round((time.time() - t0) * 1000, 2)
                print(f'[CLOUD] raw={out_text!r} action={action} latency={elapsed_ms}ms')
                return {
                    "action": action,
                    "model": f"cloud:{CLOUD_MODEL}",
                    "raw": out_text,
                    "latency_ms": elapsed_ms,
                }

            raise HTTPException(status_code=500, detail=f"Unknown backend: {BACKEND}")

    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Model request failed: {e}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Model returned non-JSON: {e}")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Model output parse error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


if __name__ == "__main__":
    def parse_args() -> argparse.Namespace:
        # 命令行参数用于覆盖默认配置，不传则使用环境变量和代码内默认值
        parser = argparse.ArgumentParser(description="VLA Server")
        parser.add_argument("--backend", default=BACKEND, choices=["rule", "local-model", "ollama", "cloud"], help="执行后端")
        parser.add_argument("--host", default=os.getenv("VLA_HOST", "0.0.0.0"), help="监听地址")
        parser.add_argument("--port", type=int, default=int(os.getenv("VLA_PORT", "8000")), help="监听端口")
        # local-model 参数（保持历史默认值）
        parser.add_argument("--local-model-dir", default=LOCAL_MODEL_DIR, help="本地模型目录")
        parser.add_argument("--local-max-new-tokens", type=int, default=LOCAL_MAX_NEW_TOKENS, help="本地模型最大新 token")
        parser.add_argument("--local-top-p", type=float, default=LOCAL_TOP_P, help="本地模型 top_p")
        parser.add_argument("--local-temperature", type=float, default=LOCAL_TEMPERATURE, help="本地模型温度")
        # ollama 参数
        parser.add_argument("--ollama-url", default=OLLAMA_URL, help="ollama 服务地址")
        parser.add_argument("--ollama-model", default=OLLAMA_MODEL, help="ollama 模型名")
        parser.add_argument("--ollama-timeout-s", type=float, default=OLLAMA_TIMEOUT_S, help="ollama 超时秒数")
        # cloud 参数
        parser.add_argument("--cloud-base-url", default=CLOUD_BASE_URL, help="cloud 接口 base_url")
        parser.add_argument("--cloud-api-key", default=CLOUD_API_KEY, help="cloud 鉴权 Key")
        parser.add_argument("--cloud-model", default=CLOUD_MODEL, help="cloud 模型名")
        parser.add_argument("--cloud-timeout-s", type=float, default=CLOUD_TIMEOUT_S, help="cloud 超时秒数")
        parser.add_argument("--cloud-max-tokens", type=int, default=CLOUD_MAX_TOKENS, help="cloud max_tokens")
        parser.add_argument("--cloud-temperature", type=float, default=CLOUD_TEMPERATURE, help="cloud temperature")
        return parser.parse_args()

    def apply_cli_config(args: argparse.Namespace) -> None:
        # CLI 配置生效于启动前的全局变量，用于后续所有推理逻辑
        global BACKEND
        global LOCAL_MODEL_DIR, LOCAL_MAX_NEW_TOKENS, LOCAL_TOP_P, LOCAL_TEMPERATURE
        global OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_S
        global CLOUD_BASE_URL, CLOUD_API_KEY, CLOUD_MODEL, CLOUD_TIMEOUT_S, CLOUD_MAX_TOKENS, CLOUD_TEMPERATURE

        BACKEND = args.backend
        LOCAL_MODEL_DIR = args.local_model_dir
        LOCAL_MAX_NEW_TOKENS = args.local_max_new_tokens
        LOCAL_TOP_P = args.local_top_p
        LOCAL_TEMPERATURE = args.local_temperature
        OLLAMA_URL = args.ollama_url.rstrip("/")
        OLLAMA_MODEL = args.ollama_model
        OLLAMA_TIMEOUT_S = args.ollama_timeout_s
        CLOUD_BASE_URL = args.cloud_base_url.rstrip("/")
        CLOUD_API_KEY = args.cloud_api_key
        CLOUD_MODEL = args.cloud_model
        CLOUD_TIMEOUT_S = args.cloud_timeout_s
        CLOUD_MAX_TOKENS = args.cloud_max_tokens
        CLOUD_TEMPERATURE = args.cloud_temperature

    args = parse_args()
    apply_cli_config(args)
    uvicorn.run(app, host=args.host, port=args.port)
