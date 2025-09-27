"""Compatibility helpers for newer pymunk versions."""

from __future__ import annotations

from functools import lru_cache


def ensure_add_collision_handler() -> None:
    """Add ``Space.add_collision_handler`` if pymunk removed it.

    ``gym-pusht`` expects the legacy PyMunk API returning a handler object. Newer
    PyMunk releases only expose :meth:`Space.on_collision`.
    """
    try:
        import pymunk  # type: ignore
        from pymunk import space as space_module  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - hard failure at import time
        raise RuntimeError("pymunk must be installed to use gym-pusht data generation") from exc

    if hasattr(space_module.Space, "add_collision_handler"):
        return

    ffi = space_module.ffi
    lib = space_module.lib
    CollisionHandler = space_module.CollisionHandler

    def add_collision_handler(self, collision_type_a, collision_type_b):  # type: ignore[override]
        key = (collision_type_a, collision_type_b)
        handler = self._handlers.get(key)
        if handler is not None:
            handler.data.setdefault("post_solve", None)
            handler.data.setdefault("begin", None)
            handler.data.setdefault("pre_solve", None)
            handler.data.setdefault("separate", None)
            return handler

        wildcard = int(ffi.cast("uintptr_t", ~0))
        a = wildcard if collision_type_a is None else collision_type_a
        b = wildcard if collision_type_b is None else collision_type_b
        c_handler = lib.cpSpaceAddCollisionHandler(self._space, a, b)
        handler = CollisionHandler(c_handler, self)
        handler.data.setdefault("post_solve", None)
        handler.data.setdefault("begin", None)
        handler.data.setdefault("pre_solve", None)
        handler.data.setdefault("separate", None)
        self._handlers[key] = handler
        return handler

    space_module.Space.add_collision_handler = add_collision_handler  # type: ignore[attr-defined]


@lru_cache(maxsize=1)
def patched() -> bool:
    """Ensure the patch is applied once and report whether it was necessary."""
    try:
        import pymunk  # noqa: F401
    except ModuleNotFoundError:
        return False
    has_attr = hasattr(pymunk.Space, "add_collision_handler")
    ensure_add_collision_handler()
    return not has_attr


__all__ = ["ensure_add_collision_handler", "patched"]
