from __future__ import annotations

from dataclasses import dataclass, field

from .core import Capability


@dataclass
class MockBackend:
    name: str = "mock"
    capabilities: frozenset[Capability] = frozenset(Capability)
    events: list[str] = field(default_factory=list)

    def connect(self) -> None:
        self.events.append("connect")

    def close(self) -> None:
        self.events.append("close")

    def move(self, linear: float, angular: float, duration_s: float) -> None:
        self.events.append(f"move linear={linear:.2f} angular={angular:.2f} duration={duration_s:.2f}")

    def stop(self) -> None:
        self.events.append("stop")

    def show_expression(self, name: str) -> None:
        self.events.append(f"expression {name}")

    def speak(self, text: str) -> None:
        self.events.append(f"speech {text}")
