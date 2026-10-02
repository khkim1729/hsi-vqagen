from hsi_vqagen.inference.telemetry import PeakMemorySampler, peak_vram_measurement
from hsi_vqagen.utils.logging import redact_secrets


def test_peak_vram_null_records_reason() -> None:
    measurement = peak_vram_measurement(reader=lambda: None, method="test-reader")
    assert measurement.peak_vram_bytes is None
    assert measurement.method == "test-reader"
    assert measurement.unavailable_reason


def test_peak_vram_value_records_method() -> None:
    measurement = peak_vram_measurement(reader=lambda: 4096, method="torch.cuda")
    assert measurement.peak_vram_bytes == 4096
    assert measurement.method == "torch.cuda"
    assert measurement.unavailable_reason is None


def test_redacts_tokens_and_authorization_headers() -> None:
    fake_tokens = [
        "h" + "f_" + "abcdefghijklmnopqrstuvwxyz",
        "g" + "hp_" + "abcdefghijklmnopqrstuvwxyz",
        "o" + "lp_" + "abcdefghijklmnopqrstuvwxyz",
    ]
    text = (
        "Authorization: Bearer abc.def.ghi "
        + " ".join(fake_tokens)
    )
    redacted = redact_secrets(text)
    assert "abc.def.ghi" not in redacted
    assert all(token not in redacted for token in fake_tokens)
    assert redacted.count("[REDACTED]") >= 4


def test_peak_sampler_keeps_maximum_observed_value() -> None:
    values = iter((100, 500, 300))
    sampler = PeakMemorySampler(lambda: next(values), interval_seconds=60)
    with sampler:
        sampler._sample()
    assert sampler.peak_bytes == 500
