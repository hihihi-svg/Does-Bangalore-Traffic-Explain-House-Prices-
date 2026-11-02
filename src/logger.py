"""
Server and Model Training Logging Subsystem
============================================
Provides unified, thread-safe server logging with dual output:
1. Formatted console stream with colorized prefixes
2. Persistent server logs in `logs/server.log` and `logs/training.log`
Includes system resource telemetry (RAM, CPU, PyTorch GPU/CPU device).
"""

import os
import sys
import logging
import datetime
import psutil

os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

class ServerLogger:
    _instance = None

    def __init__(self, name="BangaloreHousingGNN", log_dir="logs"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.log_dir = log_dir
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            # 1. Server Master Log (Rotating / Persistent)
            server_log_file = os.path.join(log_dir, "server.log")
            fh_server = logging.FileHandler(server_log_file, mode="a", encoding="utf-8")
            fh_server.setLevel(logging.INFO)
            fmt_file = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            fh_server.setFormatter(fmt_file)
            self.logger.addHandler(fh_server)
            
            # 2. Training Dedicated Log
            train_log_file = os.path.join(log_dir, "training.log")
            fh_train = logging.FileHandler(train_log_file, mode="a", encoding="utf-8")
            fh_train.setLevel(logging.DEBUG)
            fh_train.setFormatter(fmt_file)
            self.logger.addHandler(fh_train)
            
            # 3. Clean Console Stream Handler
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(logging.INFO)
            fmt_console = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S"
            )
            ch.setFormatter(fmt_console)
            self.logger.addHandler(ch)

    @classmethod
    def get_logger(cls, name="BangaloreHousingGNN"):
        if cls._instance is None:
            cls._instance = ServerLogger(name)
        return cls._instance.logger

def log_system_telemetry():
    logger = ServerLogger.get_logger()
    mem = psutil.virtual_memory()
    logger.info("=" * 80)
    logger.info("SYSTEM & SERVER TELEMETRY")
    logger.info(f"OS: {sys.platform} | Python: {sys.version.split()[0]}")
    logger.info(f"Host Memory: Total={mem.total / (1024**3):.2f} GB | Available={mem.available / (1024**3):.2f} GB ({mem.percent}% used)")
    logger.info(f"CPU Physical Cores: {psutil.cpu_count(logical=False)} | Logical Threads: {psutil.cpu_count(logical=True)}")
    logger.info("=" * 80)

if __name__ == "__main__":
    logger = ServerLogger.get_logger()
    logger.info("Server logger initialized successfully.")
    log_system_telemetry()
