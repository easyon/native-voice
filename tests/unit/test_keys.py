from __future__ import annotations

import time
from pathlib import Path

from native_voice.security.keys import generate_key, hash_secret, parse_token, verify_secret
from native_voice.security.keystore import KeyStore


def test_key_format_and_hash():
    key_id, secret, token = generate_key()
    assert token.startswith("nv_")
    assert parse_token(token) == (key_id, secret)
    assert parse_token("bad") is None
    encoded = hash_secret(secret)
    assert verify_secret(secret, encoded)
    assert not verify_secret("wrong", encoded)


def test_keystore_verify_and_disable(tmp_path: Path):
    store = KeyStore(tmp_path / "k.db")
    record, token = store.create_key("unit", ["asr", "tts"])
    assert record.key_id in token
    got = store.verify_token(token)
    assert got is not None and got.scopes == ("asr", "tts")
    assert store.verify_token(token + "x") is None
    store.set_enabled(record.key_id, False)
    assert store.verify_token(token) is None
    store.set_enabled(record.key_id, True)
    assert store.verify_token(token) is not None
    store.close()


def test_keystore_expiry(tmp_path: Path):
    store = KeyStore(tmp_path / "k.db")
    _, token = store.create_key("exp", ["asr"], expires_at=time.time() - 1)
    assert store.verify_token(token) is None
    store.close()
