from unittest.mock import patch

import redis

from app.cache import (
    _hash,
    get_cached_embedding,
    get_cached_search,
    set_cached_embedding,
    set_cached_search,
)


def test_hash_is_deterministic_and_fixed_length():
    assert _hash("hello world") == _hash("hello world")
    assert _hash("hello world") != _hash("goodbye world")
    assert len(_hash("hello world")) == 24


def test_cache_degrades_gracefully_when_redis_unreachable():
    """The cache is a speed optimization, not a hard dependency — if Redis
    is down, search should still work, just uncached. None of these should
    raise.
    """
    with patch("app.cache.get_client", side_effect=redis.RedisError("down")):
        assert get_cached_search("semantic", "q", 10) is None
        set_cached_search("semantic", "q", 10, {"ok": True})  # must not raise
        assert get_cached_embedding("local", "q") is None
        set_cached_embedding("local", "q", [0.1, 0.2])  # must not raise
