"""novare/channels/registry.py — 渠道自动发现

扫描 novare.channels 包下的所有模块，找到 BaseChannel 子类。
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from novare.channels.base import BaseChannel

logger = logging.getLogger("novare.channels")

_INTERNAL = frozenset({"base", "manager", "registry", "adapter", "events", "bus"})


def discover_channel_names() -> list[str]:
    """扫描 novare.channels 包，返回所有非内部模块名（零导入）。"""
    import novare.channels as pkg

    return [
        name
        for _, name, ispkg in pkgutil.iter_modules(pkg.__path__)
        if name not in _INTERNAL and not ispkg
    ]


def load_channel_class(module_name: str) -> type[BaseChannel]:
    """导入模块并返回第一个 BaseChannel 子类。"""
    from novare.channels.base import BaseChannel as _Base

    mod = importlib.import_module(f"novare.channels.{module_name}")
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and issubclass(obj, _Base) and obj is not _Base:
            return obj
    raise ImportError(f"No BaseChannel subclass in novare.channels.{module_name}")


def discover_all() -> dict[str, type[BaseChannel]]:
    """发现所有内置渠道。"""
    builtin: dict[str, type[BaseChannel]] = {}
    for modname in discover_channel_names():
        try:
            builtin[modname] = load_channel_class(modname)
        except ImportError as e:
            logger.debug("Skipping built-in channel '%s': %s", modname, e)
    return builtin
