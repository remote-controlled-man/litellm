"""Atomic affinity claims shared by deployment and tier-model selection."""

from collections.abc import Mapping
from typing import Final

from litellm._logging import verbose_router_logger
from litellm.caching.redis_batch import Batch
from litellm.redis.stores.router_session_pins import ROUTER_SESSION_PINS_TARGET, PinKey, RouterSessionPinStore

__all__ = [
    "ROUTER_SESSION_PINS_TARGET",
    "claim_affinity_pin",
    "claim_affinity_pin_in_memory",
    "read_affinity_pin",
    "set_local_affinity_pin",
]


def set_local_affinity_pin(pins: RouterSessionPinStore, cache_key: PinKey, value: object, ttl_seconds: int) -> None:
    pins.set_local(cache_key, value, ttl_seconds)


def _legacy_pin_matches(stored: object, pin_value: Mapping[str, str]) -> bool:
    if isinstance(stored, dict):
        return all(stored.get(key) is not None and str(stored[key]) == value for key, value in pin_value.items())
    return isinstance(stored, str) and len(pin_value) == 1 and stored in pin_value.values()


def claim_affinity_pin_in_memory(
    pins: RouterSessionPinStore,
    cache_key: PinKey,
    pin_value: Mapping[str, str],
    ttl_seconds: int,
    *,
    eligible_values: tuple[Mapping[str, str], ...] | None = None,
    preserve_legacy_model_ids: bool = False,
) -> object:
    """No await between read and write, so same-loop claims agree during a Redis outage."""
    existing: Final = pins.get_local(cache_key)
    if existing is not None and eligible_values is None:
        if _legacy_pin_matches(existing, pin_value):
            set_local_affinity_pin(pins, cache_key, pin_value, ttl_seconds)
        return existing
    eligible_existing: Final = existing in (eligible_values or ()) or (
        preserve_legacy_model_ids
        and isinstance(existing, str)
        and any("model_id" in value and _legacy_pin_matches(existing, value) for value in (eligible_values or ()))
    )
    winner: Final = existing if existing is not None and eligible_existing else pin_value
    set_local_affinity_pin(pins, cache_key, winner, ttl_seconds)
    return winner


async def read_affinity_pin(pins: RouterSessionPinStore, cache_key: PinKey, ttl_seconds: int) -> object:
    """Refresh the authoritative pin without writing a previously observed value back."""
    if pins.has_redis:
        try:
            value: Final = await pins.read(cache_key, ttl_seconds, batch=Batch.PRE_CALL)
        except Exception as error:  # noqa: BLE001  # Redis faults retain pod-local continuity, unlike a successful miss
            verbose_router_logger.debug("Affinity cache: Redis read failed, using pod-local pin. error=%s", error)
        else:
            return value
    local_value: Final = pins.get_local(cache_key)
    if local_value is not None:
        set_local_affinity_pin(pins, cache_key, local_value, ttl_seconds)
    return local_value


async def claim_affinity_pin(
    pins: RouterSessionPinStore,
    cache_key: PinKey,
    pin_value: Mapping[str, str],
    ttl_seconds: int,
    *,
    eligible_values: tuple[Mapping[str, str], ...] | None = None,
    preserve_legacy_model_ids: bool = False,
) -> object:
    """Return the authoritative first writer, replacing it only when it becomes ineligible.

    Eligible claims refresh the returned winner. Legacy deployment claims only refresh
    a matching candidate. Resolve Redis per call because the proxy attaches it lazily.
    """
    if pins.has_redis:
        try:
            winner: Final = await pins.claim(
                cache_key,
                pin_value,
                ttl_seconds,
                eligible_values=eligible_values,
                preserve_legacy_model_ids=preserve_legacy_model_ids,
            )
        except Exception as error:  # noqa: BLE001  # Redis/Lua faults retain same-pod affinity through local claims
            verbose_router_logger.debug("Affinity cache: Redis claim failed, using pod-local claim. error=%s", error)
        else:
            return pin_value if winner is None else winner
    return claim_affinity_pin_in_memory(
        pins,
        cache_key,
        pin_value,
        ttl_seconds,
        eligible_values=eligible_values,
        preserve_legacy_model_ids=preserve_legacy_model_ids,
    )
