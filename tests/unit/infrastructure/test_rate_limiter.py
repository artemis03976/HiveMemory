from hivememory.infrastructure.rate_limiter import RateLimiter


def test_allow_consumes_initial_tokens(monkeypatch):
    monkeypatch.setattr("hivememory.infrastructure.rate_limiter.time.time", lambda: 100.0)
    limiter = RateLimiter(max_rate=2)

    assert limiter.allow() is True
    assert limiter.allow() is True
    assert limiter.allow() is False


def test_allow_refills_tokens_over_elapsed_time(monkeypatch):
    times = iter([100.0, 100.0, 100.0, 100.5])
    monkeypatch.setattr("hivememory.infrastructure.rate_limiter.time.time", lambda: next(times))
    limiter = RateLimiter(max_rate=2)
    limiter.allow()
    limiter.allow()

    assert limiter.allow() is True


def test_reset_restores_bucket_capacity(monkeypatch):
    now = 100.0
    monkeypatch.setattr("hivememory.infrastructure.rate_limiter.time.time", lambda: now)
    limiter = RateLimiter(max_rate=1)
    limiter.allow()
    assert limiter.allow() is False

    limiter.reset()

    assert limiter.allow() is True
