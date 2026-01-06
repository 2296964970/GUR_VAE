from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

from .errors import ValidationError

T = TypeVar("T")


def reject_unknown_keys(
    d: Mapping[str, Any],
    *,
    allowed: Iterable[str],
    path: str,
    exc: type[ValidationError] = ValidationError,
    kind: str = "config",
    prefix: str = "[error]",
) -> None:
    unknown = sorted(set(d.keys()) - set(allowed))
    if unknown:
        raise exc(f"{prefix} Unknown {kind} key(s) at {path}: {', '.join(unknown)}")


def require_mapping(
    value: Any,
    *,
    err: str,
    exc: type[ValidationError] = ValidationError,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise exc(err)
    return value


def require_non_empty_str(
    value: Any,
    *,
    err: str,
    exc: type[ValidationError] = ValidationError,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise exc(err)
    return value.strip()


def require_bool(
    value: Any,
    *,
    err: str,
    exc: type[ValidationError] = ValidationError,
) -> bool:
    if not isinstance(value, bool):
        raise exc(err)
    return bool(value)


def require_int(
    value: Any,
    *,
    err: str,
    exc: type[ValidationError] = ValidationError,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise exc(err)
    return int(value)


def require_float(
    value: Any,
    *,
    err: str,
    exc: type[ValidationError] = ValidationError,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise exc(err)
    return float(value)


def require_list(
    value: Any,
    *,
    err: str,
    exc: type[ValidationError] = ValidationError,
) -> list:
    if not isinstance(value, list):
        raise exc(err)
    return value


def require_sequence(
    value: Any,
    *,
    err: str,
    exc: type[ValidationError] = ValidationError,
) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise exc(err)
    return value


def _require_tuple(
    value: Any,
    *,
    err: str,
    empty_err: str | None,
    exc: type[ValidationError],
    parse_item: Callable[[Any, int], T],
) -> tuple[T, ...]:
    seq = require_sequence(value, err=err, exc=exc)
    if not seq:
        raise exc(empty_err or err)
    out: list[T] = []
    for i, item in enumerate(seq):
        out.append(parse_item(item, i))
    return tuple(out)


def _require_range2(
    items: Sequence[T],
    *,
    err: str,
    len_err: str | None,
    range_err: str | None,
    exc: type[ValidationError],
) -> tuple[T, T]:
    if len(items) != 2:
        raise exc(len_err or err)
    lo, hi = items[0], items[1]
    if lo > hi:
        raise exc(range_err or err)
    return lo, hi


def require_int_tuple(
    value: Any,
    *,
    err: str,
    empty_err: str | None = None,
    item_err: Callable[[int], str] | None = None,
    exc: type[ValidationError] = ValidationError,
) -> tuple[int, ...]:
    def _parse(item: Any, i: int) -> int:
        return require_int(item, err=item_err(i) if item_err else err, exc=exc)

    return _require_tuple(
        value,
        err=err,
        empty_err=empty_err,
        exc=exc,
        parse_item=_parse,
    )


def require_float_tuple(
    value: Any,
    *,
    err: str,
    empty_err: str | None = None,
    item_err: Callable[[int], str] | None = None,
    exc: type[ValidationError] = ValidationError,
) -> tuple[float, ...]:
    def _parse(item: Any, i: int) -> float:
        return require_float(item, err=item_err(i) if item_err else err, exc=exc)

    return _require_tuple(
        value,
        err=err,
        empty_err=empty_err,
        exc=exc,
        parse_item=_parse,
    )


def require_range2_int(
    value: Any,
    *,
    err: str,
    empty_err: str | None = None,
    len_err: str | None = None,
    range_err: str | None = None,
    item_err: Callable[[int], str] | None = None,
    exc: type[ValidationError] = ValidationError,
) -> tuple[int, int]:
    items = require_int_tuple(
        value,
        err=err,
        empty_err=empty_err,
        item_err=item_err,
        exc=exc,
    )
    lo, hi = _require_range2(items, err=err, len_err=len_err, range_err=range_err, exc=exc)
    return int(lo), int(hi)


def require_range2_float(
    value: Any,
    *,
    err: str,
    empty_err: str | None = None,
    len_err: str | None = None,
    range_err: str | None = None,
    item_err: Callable[[int], str] | None = None,
    exc: type[ValidationError] = ValidationError,
) -> tuple[float, float]:
    items = require_float_tuple(
        value,
        err=err,
        empty_err=empty_err,
        item_err=item_err,
        exc=exc,
    )
    lo, hi = _require_range2(items, err=err, len_err=len_err, range_err=range_err, exc=exc)
    return float(lo), float(hi)


__all__ = [
    "reject_unknown_keys",
    "require_mapping",
    "require_non_empty_str",
    "require_bool",
    "require_int",
    "require_float",
    "require_list",
    "require_sequence",
    "require_int_tuple",
    "require_float_tuple",
    "require_range2_int",
    "require_range2_float",
]
