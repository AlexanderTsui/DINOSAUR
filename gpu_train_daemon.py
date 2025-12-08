"""
GPU训练守护进程脚本

功能：
1. 任务队列管理 - 按顺序排队训练任务
2. 多GPU轮询 - 同时监控多张卡，哪张空闲用哪张
3. 文件锁抢卡 - 多用户/多进程互斥，避免同时抢同一块卡
4. 自动启动训练 - GPU空闲且抢到锁时启动train_3d_mask3d.py

使用方法：
1. 修改下方 TASK_QUEUE 配置你的任务列表
2. 直接运行此脚本：python gpu_train_daemon.py
3. 脚本会在后台轮询GPU状态，空闲时自动启动训练
"""

import os
import sys
import time
import fcntl
import subprocess
import signal
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# ==================== 配置区 ====================

# 任务队列：每个任务是一个字典
# gpu_ids: 候选GPU列表，按优先级排序，哪张空闲用哪张
# 按顺序执行，前一个任务完成后才执行下一个
TASK_QUEUE = [
    {
        "name": "恢复训练-07-epoch030",
        "gpu_ids": [0,1,2,3,4,5,6,7],
        "config": "config/config_train_mask3d.yaml",
        "extra_args": [
            "--resume",
            "/home/pbw/data1/3D_PointCloud_Segmentation/PLSG_Net/Model_Code/src/DINOSAUR/checkpoints_mask3d/07/epoch_030.pth"
        ],
    },
    # {
    #     "name": "训练任务1",
    #     "gpu_ids": [0,1,2,3,4,5,6,7],            # 候选GPU列表（按优先级），哪张空闲用哪张
    #     "config": "config/config_train_mask3d.yaml",
    #     "extra_args": [],
    # },
    # {
    #     "name": "训练任务2",
    #     "gpu_ids": [5,6,7],            # 同样支持多卡
    #     "config": "config/config_train_mask3d_copy.yaml",
    #     "extra_args": [],
    # },
]

# 轮询间隔（秒）
POLL_INTERVAL = 10

# 锁文件目录（改到用户目录，避免/tmp空间不足）
LOCK_DIR = os.path.expanduser("~/data/cache/.gpu_train_locks")

# 日志文件（None表示不写文件，只打印到终端）
LOG_FILE = None  # 或设置为 "/path/to/daemon.log"

# Conda环境名称
CONDA_ENV = "CloudPoints"

# 训练脚本名称
TRAIN_SCRIPT = "train_3d_mask3d.py"

# GPU空闲判定配置
# 方式1：基于已用显存 - 显存使用低于此值(MB)视为空闲
GPU_MEM_USED_THRESHOLD_MB = 2000  # 已用显存<2GB时可用

# 方式2：基于剩余显存 - 剩余显存高于此值(MB)视为可用（推荐）
GPU_MEM_FREE_THRESHOLD_MB = 23000 # 剩余>20GB可用

# 判断模式："used"基于已用显存，"free"基于剩余显存
GPU_CHECK_MODE = "free"  # 推荐用"free"模式

# ==================== 工具函数 ====================

def log(msg: str):
    """打印日志，带时间戳"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    sys.stdout.flush()
    
    if LOG_FILE:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")


def get_gpu_status(gpu_id: int) -> dict:
    """
    查询指定GPU的状态
    返回: {"is_available": bool, "memory_used_mb": int, "memory_free_mb": int, 
           "memory_total_mb": int, "processes": list, "reason": str}
    """
    try:
        # 查询显存：已用、空闲、总量
        cmd_mem = [
            "nvidia-smi",
            "-i", str(gpu_id),
            "--query-gpu=memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits"
        ]
        result_mem = subprocess.run(cmd_mem, capture_output=True, text=True, timeout=10)
        
        if result_mem.returncode == 0:
            parts = result_mem.stdout.strip().split(",")
            memory_used = int(parts[0].strip())
            memory_free = int(parts[1].strip())
            memory_total = int(parts[2].strip())
        else:
            memory_used, memory_free, memory_total = -1, -1, -1
        
        # 查询进程列表（仅用于日志显示）
        cmd_proc = [
            "nvidia-smi",
            "-i", str(gpu_id),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader"
        ]
        result_proc = subprocess.run(cmd_proc, capture_output=True, text=True, timeout=10)
        
        processes = []
        if result_proc.returncode == 0 and result_proc.stdout.strip():
            for line in result_proc.stdout.strip().split("\n"):
                if line.strip():
                    processes.append(line.strip())
        
        # 判断是否可用
        is_available = False
        reason = ""
        
        if memory_used < 0 or memory_free < 0:
            reason = "无法获取显存信息"
        elif GPU_CHECK_MODE == "free":
            if memory_free >= GPU_MEM_FREE_THRESHOLD_MB:
                is_available = True
                reason = f"剩余{memory_free}MB >= {GPU_MEM_FREE_THRESHOLD_MB}MB"
            else:
                reason = f"剩余{memory_free}MB < {GPU_MEM_FREE_THRESHOLD_MB}MB"
        else:
            if memory_used <= GPU_MEM_USED_THRESHOLD_MB:
                is_available = True
                reason = f"已用{memory_used}MB <= {GPU_MEM_USED_THRESHOLD_MB}MB"
            else:
                reason = f"已用{memory_used}MB > {GPU_MEM_USED_THRESHOLD_MB}MB"
        
        return {
            "is_available": is_available,
            "memory_used_mb": memory_used,
            "memory_free_mb": memory_free,
            "memory_total_mb": memory_total,
            "processes": processes,
            "reason": reason
        }
    except subprocess.TimeoutExpired:
        return {"is_available": False, "memory_used_mb": -1, "memory_free_mb": -1, 
                "memory_total_mb": -1, "processes": [], "reason": "nvidia-smi超时"}
    except Exception as e:
        return {"is_available": False, "memory_used_mb": -1, "memory_free_mb": -1,
                "memory_total_mb": -1, "processes": [], "reason": str(e)}


def get_all_gpu_status(gpu_ids: List[int]) -> dict:
    """批量查询多张GPU状态，返回 {gpu_id: status_dict}"""
    result = {}
    for gpu_id in gpu_ids:
        result[gpu_id] = get_gpu_status(gpu_id)
    return result


class GPULock:
    """GPU文件锁，用于多进程/多用户互斥抢卡"""
    
    def __init__(self, gpu_id: int, lock_dir: str = LOCK_DIR):
        self.gpu_id = gpu_id
        self.lock_dir = Path(lock_dir)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.lock_file = self.lock_dir / f"gpu_{gpu_id}.lock"
        self.fd = None
        self.locked = False
    
    def try_acquire(self) -> bool:
        """尝试获取锁（非阻塞）"""
        try:
            self.fd = open(self.lock_file, "w")
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fd.write(f"pid={os.getpid()}\n")
            self.fd.write(f"time={datetime.now().isoformat()}\n")
            self.fd.flush()
            self.locked = True
            return True
        except (IOError, OSError):
            if self.fd:
                self.fd.close()
                self.fd = None
            return False
    
    def release(self):
        """释放锁"""
        if self.locked and self.fd:
            try:
                fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
                self.fd.close()
                self.fd = None
                self.locked = False
                if self.lock_file.exists():
                    self.lock_file.unlink()
            except Exception as e:
                log(f"[警告] 释放锁失败: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def find_available_gpu(gpu_ids: List[int], verbose: bool = True) -> Tuple[Optional[int], Optional[GPULock], dict]:
    """
    在候选GPU列表中查找第一个可用的GPU并尝试获取锁
    
    返回: (gpu_id, lock, status) 或 (None, None, all_status)
        - 成功时返回 (可用的gpu_id, 获取的锁对象, 该GPU的状态)
        - 失败时返回 (None, None, 所有GPU的状态汇总)
    """
    all_status = get_all_gpu_status(gpu_ids)
    
    for gpu_id in gpu_ids:
        status = all_status[gpu_id]
        
        if not status["is_available"]:
            continue
        
        # GPU显存可用，尝试获取锁
        lock = GPULock(gpu_id)
        if not lock.try_acquire():
            if verbose:
                log(f"  GPU {gpu_id} 显存可用但锁被占用")
            continue
        
        # 再次确认（防止获取锁期间状态变化）
        status = get_gpu_status(gpu_id)
        if not status["is_available"]:
            lock.release()
            if verbose:
                log(f"  GPU {gpu_id} 获取锁后状态变化，释放")
            continue
        
        # 成功找到可用GPU并获取锁
        return gpu_id, lock, status
    
    return None, None, all_status


def run_training(task: dict, gpu_id: int, script_dir: str) -> int:
    """
    运行训练任务
    
    参数:
        task: 任务配置字典
        gpu_id: 实际使用的GPU ID
        script_dir: 训练脚本所在目录
    
    返回: 进程退出码
    """
    config = task["config"]
    extra_args = task.get("extra_args", [])
    task_name = task.get("name", "unnamed")
    
    train_script_path = os.path.join(script_dir, TRAIN_SCRIPT)
    config_path = os.path.join(script_dir, config)
    
    cmd = [
        "conda", "run", "-n", CONDA_ENV, "--no-capture-output",
        "python", train_script_path,
        "--config", config_path
    ] + extra_args
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    log(f"启动训练: {task_name}")
    log(f"  GPU: {gpu_id}")
    log(f"  配置: {config}")
    log(f"  命令: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            cwd=script_dir,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        
        return_code = process.wait()
        
        elapsed = time.time() - start_time
        elapsed_str = f"{elapsed/3600:.2f}小时" if elapsed > 3600 else f"{elapsed/60:.1f}分钟"
        
        if return_code == 0:
            log(f"✓ 训练完成: {task_name} (GPU:{gpu_id}, 耗时: {elapsed_str})")
        else:
            log(f"✗ 训练失败: {task_name} (GPU:{gpu_id}, 返回码: {return_code}, 耗时: {elapsed_str})")
        
        return return_code
        
    except KeyboardInterrupt:
        log(f"训练被用户中断: {task_name}")
        process.terminate()
        process.wait()
        return -1
    except Exception as e:
        log(f"训练异常: {task_name}, 错误: {e}")
        return -2


def format_gpu_summary(gpu_ids: List[int], all_status: dict) -> str:
    """格式化GPU状态摘要"""
    parts = []
    for gpu_id in gpu_ids:
        s = all_status.get(gpu_id, {})
        used = s.get("memory_used_mb", -1)
        free = s.get("memory_free_mb", -1)
        avail = "✓" if s.get("is_available", False) else "✗"
        parts.append(f"GPU{gpu_id}:{avail}({used}/{free}MB)")
    return " | ".join(parts)


def wait_for_gpu_and_run(task: dict, script_dir: str) -> int:
    """
    等待任意一张候选GPU可用，获取锁后运行训练
    
    返回: 训练进程退出码
    """
    gpu_ids = task.get("gpu_ids", [task.get("gpu_id", 0)])  # 兼容旧配置
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    
    task_name = task.get("name", "unnamed")
    
    log(f"等待GPU可用: {task_name}")
    log(f"  候选GPU: {gpu_ids}")
    log(f"  判断模式: {GPU_CHECK_MODE}, 阈值: " + 
        (f"剩余>{GPU_MEM_FREE_THRESHOLD_MB}MB" if GPU_CHECK_MODE == "free" 
         else f"已用<{GPU_MEM_USED_THRESHOLD_MB}MB"))
    
    poll_count = 0
    while True:
        poll_count += 1
        
        # 查找可用GPU
        gpu_id, lock, status_or_all = find_available_gpu(gpu_ids, verbose=False)
        
        if gpu_id is not None and lock is not None:
            # 找到可用GPU
            try:
                log(f"✓ 获取到GPU {gpu_id}")
                log(f"  显存: 已用 {status_or_all['memory_used_mb']}MB / "
                    f"剩余 {status_or_all['memory_free_mb']}MB / "
                    f"总计 {status_or_all['memory_total_mb']}MB")
                
                return_code = run_training(task, gpu_id, script_dir)
                return return_code
            finally:
                lock.release()
                log(f"已释放GPU {gpu_id} 锁")
        else:
            # 所有GPU都不可用，打印状态并等待
            if poll_count == 1 or poll_count % 6 == 0:  # 首次和每分钟打印一次
                summary = format_gpu_summary(gpu_ids, status_or_all)
                log(f"  等待中... {summary}")
            
            time.sleep(POLL_INTERVAL)


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    log("=" * 60)
    log("GPU训练守护进程启动（多卡轮询模式）")
    log("=" * 60)
    log(f"脚本目录: {script_dir}")
    log(f"Conda环境: {CONDA_ENV}")
    log(f"轮询间隔: {POLL_INTERVAL}秒")
    log(f"锁文件目录: {LOCK_DIR}")
    log(f"任务数量: {len(TASK_QUEUE)}")
    
    # 显示任务队列
    log("\n任务队列:")
    for i, task in enumerate(TASK_QUEUE):
        gpu_ids = task.get("gpu_ids", [task.get("gpu_id", "?")])
        log(f"  {i+1}. {task.get('name', 'unnamed')} - 候选GPU:{gpu_ids} - {task['config']}")
    log("")
    
    # 处理信号
    def signal_handler(signum, frame):
        log("\n收到退出信号，正在退出...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 执行任务队列
    completed = 0
    failed = 0
    
    for i, task in enumerate(TASK_QUEUE):
        log(f"\n{'='*60}")
        log(f"处理任务 {i+1}/{len(TASK_QUEUE)}: {task.get('name', 'unnamed')}")
        log(f"{'='*60}")
        
        return_code = wait_for_gpu_and_run(task, script_dir)
        
        if return_code == 0:
            completed += 1
        else:
            failed += 1
            log(f"[警告] 任务失败，继续执行下一个任务")
    
    log(f"\n{'='*60}")
    log("所有任务处理完毕")
    log(f"  完成: {completed}")
    log(f"  失败: {failed}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
