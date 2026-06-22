import sys
from collections.abc import Callable
from importlib import import_module
from pathlib import Path

from engine.policy import Policy, RandomPolicy
from policies.example_policy import ExamplePolicy

PolicyFactory = Callable[[], Policy]

POLICIES: dict[str, PolicyFactory] = {
    "random": lambda: RandomPolicy(seed=0),
    "example": lambda: ExamplePolicy(seed=0),
}

def register(name: str, factory: PolicyFactory) -> None:
    POLICIES[name] = factory

def available() -> list[str]:
    return sorted(POLICIES)

def get_policy(name: str) -> Policy:
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; registered: {available()}")
    return POLICIES[name]()

_HARNESS_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("solution", "reference"):
    if (_HARNESS_ROOT / _pkg / "__init__.py").exists():
        try:
            import_module(_pkg)
        except Exception as exc:  # noqa: BLE001 — one broken add-on must not kill the harness
            print(
                f"[policies] could not import optional package {_pkg!r}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
