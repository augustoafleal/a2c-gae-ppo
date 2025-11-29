import csv
from datetime import datetime
import os

"""

class Logger:
    def __init__(self, episode_filename=None, update_filename=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if episode_filename is None:
            episode_filename = f"logs/episodes_{timestamp}.csv"
        episode_dir = os.path.dirname(episode_filename)
        if episode_dir and not os.path.exists(episode_dir):
            os.makedirs(episode_dir, exist_ok=True)
        self.episode_filename = episode_filename
        self._init_episode_file()

        if update_filename is None:
            update_filename = f"logs/updates_{timestamp}.csv"
        update_dir = os.path.dirname(update_filename)
        if update_dir and not os.path.exists(update_dir):
            os.makedirs(update_dir, exist_ok=True)
        self.update_filename = update_filename
        self._init_update_file()

    def _init_episode_file(self):
        if not os.path.exists(self.episode_filename):
            with open(self.episode_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["worker", "episode", "total_steps", "total_reward", "terminated"])

    def _init_update_file(self):
        if not os.path.exists(self.update_filename):
            with open(self.update_filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["update", "critic_loss", "actor_loss", "entropy"])

    def log_episode(self, worker, episode, total_steps, total_reward, terminated):
        with open(self.episode_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([worker, episode, total_steps, total_reward, terminated])

    def log_update(self, update_num, critic_loss=None, actor_loss=None, entropy=None):
        with open(self.update_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([update_num, critic_loss, actor_loss, entropy])
"""

import csv
from datetime import datetime
import os
import psutil
import time
import GPUtil
import threading

import csv
from datetime import datetime
import os
import psutil
import GPUtil


class Logger:
    def __init__(self, episode_filename=None, update_filename=None, resources_filename=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.episode_filename = self._init_file(
            episode_filename,
            f"logs/episodes_{timestamp}.csv",
            ["worker", "episode", "total_steps", "total_reward", "terminated"],
        )

        self.update_filename = self._init_file(
            update_filename, f"logs/updates_{timestamp}.csv", ["update", "critic_loss", "actor_loss", "entropy"]
        )

        self.resources_filename = self._init_file(
            resources_filename,
            f"logs/resources_{timestamp}.csv",
            [
                "timestamp",
                "update",
                "total_steps",
                "ram_gb",
                "cpu_percent_total",
                "cpu_percent_normalized",
                "cpu_cores_used",
                "gpu_mem_gb",
                "gpu_util_percent",
                "children_processes",
                "disk_io",
            ],
        )

        self.process = psutil.Process(os.getpid())
        self.last_disk_io = self.process.io_counters() if hasattr(self.process, "io_counters") else None
        self.process.cpu_percent()  # Inicialização para leituras corretas

    def _init_file(self, filename, default_filename, headers):
        if filename is None:
            filename = default_filename

        file_dir = os.path.dirname(filename)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)

        if not os.path.exists(filename):
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        return filename

    def _get_system_resources(self):
        try:
            ram_gb = self.process.memory_info().rss / 1024**3

            cpu_percent_total = self.process.cpu_percent()
            cpu_cores = psutil.cpu_count(logical=True)
            cpu_cores_used = round(cpu_percent_total / 100, 1)
            cpu_percent_normalized = min(100.0, cpu_percent_total / cpu_cores)

            children_count = len(self.process.children(recursive=True))

            disk_io = 0
            if hasattr(self.process, "io_counters"):
                current_io = self.process.io_counters()
                if self.last_disk_io:
                    disk_io = (
                        current_io.read_bytes
                        + current_io.write_bytes
                        - self.last_disk_io.read_bytes
                        - self.last_disk_io.write_bytes
                    ) / 1024**2
                self.last_disk_io = current_io

            gpu_mem_gb = 0
            gpu_util_percent = 0
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_mem_gb = gpu.memoryUsed / 1024
                    gpu_util_percent = gpu.load * 100
            except:
                pass

            return {
                "ram_gb": round(ram_gb, 2),
                "cpu_percent_total": round(cpu_percent_total, 1),
                "cpu_percent_normalized": round(cpu_percent_normalized, 1),
                "cpu_cores_used": cpu_cores_used,
                "gpu_mem_gb": round(gpu_mem_gb, 2),
                "gpu_util_percent": round(gpu_util_percent, 1),
                "children_processes": children_count,
                "disk_io": round(disk_io, 2),
            }

        except Exception as e:
            print(f"Error fetching resources: {e}")
            return {
                "ram_gb": 0,
                "cpu_percent_total": 0,
                "cpu_percent_normalized": 0,
                "cpu_cores_used": 0,
                "gpu_mem_gb": 0,
                "gpu_util_percent": 0,
                "children_processes": 0,
                "disk_io": 0,
            }

    def log_episode(self, worker, episode, total_steps, total_reward, terminated):
        with open(self.episode_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([worker, episode, total_steps, total_reward, terminated])

    def log_update(
        self, update_num, critic_loss=None, actor_loss=None, entropy=None, total_steps=None, log_resources=True
    ):
        with open(self.update_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([update_num, critic_loss, actor_loss, entropy])

        if log_resources:
            self.log_resources(update_num, total_steps)

    def log_resources(self, update_num, total_steps=None):
        resources = self._get_system_resources()

        with open(self.resources_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    update_num,
                    total_steps or 0,
                    resources["ram_gb"],
                    resources["cpu_percent_total"],
                    resources["cpu_percent_normalized"],
                    resources["cpu_cores_used"],
                    resources["gpu_mem_gb"],
                    resources["gpu_util_percent"],
                    resources["children_processes"],
                    resources["disk_io"],
                ]
            )

        print(
            f"[RESOURCES] Update {update_num} | "
            f"RAM: {resources['ram_gb']}GB | "
            f"CPU: {resources['cpu_percent_normalized']}% ({resources['cpu_cores_used']} cores) | "
            f"GPU: {resources['gpu_mem_gb']}GB ({resources['gpu_util_percent']}%) | "
            f"Process: {resources['children_processes']}"
        )

    def log_critical_resources(self, update_num, total_steps=None):
        resources = self._get_system_resources()

        print(
            f"[CRITICAL_RESOURCES] Update {update_num} | "
            f"RAM: {resources['ram_gb']}GB | "
            f"CPU: {resources['cpu_percent_normalized']}% ({resources['cpu_cores_used']} cores) | "
            f"GPU: {resources['gpu_mem_gb']}GB | "
            f"Processos: {resources['children_processes']}"
        )

        self.log_resources(update_num, total_steps)
