from __future__ import annotations

from enum import StrEnum
from typing import Protocol


class Capability(StrEnum):
    DRIVE = "drive"
    EXPRESSION = "expression"
    SPEECH = "speech"


class CapabilityError(RuntimeError):
    pass


class RobotBackend(Protocol):
    """Small contract to implement when an official hardware SDK becomes available."""

    name: str
    capabilities: frozenset[Capability]

    def connect(self) -> None: ...

    def close(self) -> None: ...

    def move(self, linear: float, angular: float, duration_s: float) -> None: ...

    def stop(self) -> None: ...

    def show_expression(self, name: str) -> None: ...

    def speak(self, text: str) -> None: ...


class Robot:
    """Capability-gated facade used by demos."""

    def __init__(self, backend: RobotBackend) -> None:
        self.backend = backend

    def __enter__(self) -> "Robot":
        self.backend.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            self.backend.stop()
        finally:
            self.backend.close()

    def _require(self, capability: Capability) -> None:
        if capability not in self.backend.capabilities:
            raise CapabilityError(
                f"Backend {self.backend.name!r} does not expose {capability.value!r}"
            )

    def move(self, linear: float, angular: float, duration_s: float) -> None:
        self._require(Capability.DRIVE)
        if not -1.0 <= linear <= 1.0 or not -1.0 <= angular <= 1.0:
            raise ValueError("linear and angular must be normalized to [-1, 1]")
        if not 0.0 < duration_s <= 5.0:
            raise ValueError("duration_s must be in (0, 5]")
        self.backend.move(linear, angular, duration_s)

    def show_expression(self, name: str) -> None:
        self._require(Capability.EXPRESSION)
        self.backend.show_expression(name)

    def speak(self, text: str) -> None:
        self._require(Capability.SPEECH)
        self.backend.speak(text)
