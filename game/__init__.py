"""game 包：AI 单人跑团的游戏层（AGENTS.md §4）。"""

from game.agent import GameAgent
from game.engine import GameEngine
from game.script_loader import load_adventures

__all__ = ["GameAgent", "GameEngine", "load_adventures"]
