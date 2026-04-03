"""
动态记忆更新服务
监控模拟的 actions 日志文件，将新的 agent 活动实时更新到图谱
"""

import os
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from .graphiti_service import GraphitiService
from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.memory_updater')


@dataclass
class AgentActivity:
    """Agent 活动记录"""
    agent_id: str
    action: str
    timestamp: str
    platform: str

    def to_episode_text(self) -> str:
        """将活动转换为可以发送给图谱的文本描述"""
        return f"[{self.timestamp}] {self.platform}: Agent {self.agent_id} - {self.action}"


class MemoryUpdater:
    """
    动态记忆更新服务

    监控模拟的actions日志文件，将新的agent活动实时更新到图谱中。
    按批次累积活动后批量发送到图谱。

    Attributes:
        group_id: 图谱分组ID
        log_file: 监控的日志文件路径
        callback: 可选的回调函数，在发送批次后调用
    """

    # 批量发送大小
    BATCH_SIZE = 5

    # 发送间隔（秒），避免请求过快
    SEND_INTERVAL = 0.5

    # 重试配置
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # 秒

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化更新器

        Args:
            api_key: 可选的 API Key（默认从配置读取）
        """
        self.service = GraphitiService()
        self.api_key = api_key

        # 活动缓冲区
        self._activity_buffer: List[str] = []
        self._buffer_lock = threading.Lock()

        # 控制标志
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # 日志文件状态
        self.group_id: Optional[str] = None
        self.log_file: Optional[str] = None
        self.callback: Optional[Callable] = None
        self._last_read: int = 0

        # 统计
        self._total_activities = 0  # 添加到缓冲区的活动数
        self._total_sent = 0        # 成功发送到图谱的批次数
        self._total_items_sent = 0  # 成功发送到图谱的活动条数
        self._failed_count = 0      # 发送失败的批次数

        logger.info(f"MemoryUpdater 初始化完成, batch_size={self.BATCH_SIZE}")

    def start_watching(self, group_id: str, log_file: str, callback: Optional[Callable] = None):
        """
        开始监控日志文件

        Args:
            group_id: 图谱分组ID
            log_file: 日志文件路径
            callback: 可选的回调函数，在发送批次后调用
        """
        if self.running:
            logger.warning("MemoryUpdater 已经在运行中")
            return

        self.group_id = group_id
        self.log_file = log_file
        self.callback = callback
        self._last_read = 0
        self.running = True

        # 捕获当前locale
        self._locale = 'zh'

        self.thread = threading.Thread(
            target=self._watch_worker,
            daemon=True,
            name=f"MemoryUpdater-{group_id[:8]}"
        )
        self.thread.start()

        logger.info(f"MemoryUpdater 已启动: group_id={group_id}, log_file={log_file}")

    def stop_watching(self):
        """停止监控"""
        if not self.running:
            return

        self.running = False

        # 发送剩余的活动
        self._flush_remaining()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)

        logger.info(f"MemoryUpdater 已停止: group_id={self.group_id}, "
                   f"total_activities={self._total_activities}, "
                   f"batches_sent={self._total_sent}, "
                   f"items_sent={self._total_items_sent}, "
                   f"failed={self._failed_count}")

    def _watch_worker(self):
        """监控工作线程"""
        while self.running:
            try:
                # 检查文件是否存在
                if not os.path.exists(self.log_file):
                    time.sleep(1)
                    continue

                # 读取新内容
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    f.seek(self._last_read)
                    lines = f.readlines()
                    self._last_read = f.tell()

                # 处理新行
                new_activities = []
                for line in lines:
                    line = line.strip()
                    if line:
                        new_activities.append(line)

                # 添加到缓冲区
                if new_activities:
                    with self._buffer_lock:
                        self._activity_buffer.extend(new_activities)
                        self._total_activities += len(new_activities)

                    # 检查是否达到批量大小
                    if len(self._activity_buffer) >= self.BATCH_SIZE:
                        with self._buffer_lock:
                            batch = self._activity_buffer[:self.BATCH_SIZE]
                            self._activity_buffer = self._activity_buffer[self.BATCH_SIZE:]

                        self._send_batch(batch)

                        # 发送间隔，避免请求过快
                        time.sleep(self.SEND_INTERVAL)

            except FileNotFoundError:
                pass
            except Exception as e:
                logger.error(f"监控工作线程异常: {e}")

            # 睡眠间隔
            time.sleep(0.5)

    def _send_batch(self, activities: List[str]):
        """
        发送批量活动到图谱

        Args:
            activities: 活动文本列表
        """
        if not activities:
            return

        # 将多条活动合并为一条文本，用换行分隔
        combined_text = "\n".join(activities)

        # 带重试的发送
        for attempt in range(self.MAX_RETRIES):
            try:
                import asyncio
                result = asyncio.run(
                    self.service.add_episodes(self.group_id, combined_text)
                )

                if result.get("success"):
                    self._total_sent += 1
                    self._total_items_sent += len(activities)
                    logger.info(f"成功批量发送 {len(activities)} 条活动到图谱 {self.group_id}")
                    logger.debug(f"批量内容预览: {combined_text[:200]}...")

                    # 执行回调
                    if self.callback:
                        try:
                            self.callback(activities)
                        except Exception as e:
                            logger.error(f"回调函数执行失败: {e}")

                    return
                else:
                    raise Exception(result.get("error", "Unknown error"))

            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"批量发送到图谱失败 (尝试 {attempt + 1}/{self.MAX_RETRIES}): {e}")
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"批量发送到图谱失败，已重试{self.MAX_RETRIES}次: {e}")
                    self._failed_count += 1

    def _flush_remaining(self):
        """发送缓冲区中剩余的活动"""
        with self._buffer_lock:
            if self._activity_buffer:
                logger.info(f"发送剩余的 {len(self._activity_buffer)} 条活动")
                self._send_batch(self._activity_buffer)
                self._activity_buffer = []

    def add_activity(self, group_id: str, activity: AgentActivity):
        """
        添加单个活动到图谱

        Args:
            group_id: 图谱分组ID
            activity: Agent 活动记录
        """
        import asyncio
        asyncio.run(
            self.service.add_episodes(group_id, activity.to_episode_text())
        )

    def add_activity_from_dict(self, group_id: str, data: Dict[str, Any]):
        """
        从字典数据添加活动

        Args:
            group_id: 图谱分组ID
            data: 从actions.jsonl解析的字典数据
        """
        activity = AgentActivity(
            agent_id=str(data.get("agent_id", "")),
            action=str(data.get("action", data.get("action_type", ""))),
            timestamp=data.get("timestamp", ""),
            platform=str(data.get("platform", ""))
        )
        self.add_activity(group_id, activity)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._buffer_lock:
            buffer_size = len(self._activity_buffer)

        return {
            "group_id": self.group_id,
            "log_file": self.log_file,
            "batch_size": self.BATCH_SIZE,
            "total_activities": self._total_activities,
            "batches_sent": self._total_sent,
            "items_sent": self._total_items_sent,
            "failed_count": self._failed_count,
            "buffer_size": buffer_size,
            "running": self.running,
        }