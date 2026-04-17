"""
自动训练脚本：等待训练数据生成完毕后自动启动 Router 训练
"""
import time
import os
import sys
import subprocess
from pathlib import Path

TRAIN_DATA = Path("src/chunking/training_data/router_train_data.json")
VAL_DATA = Path("src/chunking/training_data/router_val_data.json")
CHECK_INTERVAL = 10  # 每10秒检查一次

def wait_for_data():
    """等待训练数据文件生成完毕"""
    print("=" * 50)
    print("等待训练数据生成...")
    print(f"  监控文件: {TRAIN_DATA}")
    print(f"  监控文件: {VAL_DATA}")
    print("=" * 50)
    
    last_size = 0
    stable_count = 0
    
    while True:
        if TRAIN_DATA.exists() and VAL_DATA.exists():
            current_size = TRAIN_DATA.stat().st_size + VAL_DATA.stat().st_size
            if current_size > 0 and current_size == last_size:
                stable_count += 1
                if stable_count >= 3:  # 文件大小连续3次不变 = 写入完毕
                    print(f"\n训练数据已就绪！")
                    print(f"  Train: {TRAIN_DATA} ({TRAIN_DATA.stat().st_size:,} bytes)")
                    print(f"  Val: {VAL_DATA} ({VAL_DATA.stat().st_size:,} bytes)")
                    return True
            else:
                stable_count = 0
            last_size = current_size
        
        time.sleep(CHECK_INTERVAL)
        print(f"  [{time.strftime('%H:%M:%S')}] 等待中...", end="")
        if TRAIN_DATA.exists():
            print(f" train={TRAIN_DATA.stat().st_size:,}B", end="")
        if VAL_DATA.exists():
            print(f" val={VAL_DATA.stat().st_size:,}B", end="")
        print()

def run_training():
    """启动 Router 训练"""
    print("\n" + "=" * 50)
    print("开始训练 Router 模型...")
    print("=" * 50 + "\n")
    
    cmd = [
        sys.executable,
        "src/chunking/train_router.py",
        "--epochs", "50",
        "--batch-size", "16",
        "--lr", "1e-4"
    ]
    
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    return result.returncode

if __name__ == "__main__":
    # 如果数据已经存在，删除旧文件等待新的
    if TRAIN_DATA.exists() and "--force" not in sys.argv:
        mod_time = TRAIN_DATA.stat().st_mtime
        age_minutes = (time.time() - mod_time) / 60
        if age_minutes < 5:
            print(f"训练数据刚生成 ({age_minutes:.1f} 分钟前)，直接开始训练...")
            exit(run_training())
    
    wait_for_data()
    exit(run_training())
