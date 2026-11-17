"""API key management command."""
from __future__ import annotations

from ..config import load_config
from ..security.keystore import KeyStore


def run(args) -> None:
    cfg = load_config(args.config)
    store = KeyStore(cfg.auth.db_path)
    try:
        if args.action == "create":
            record, token = store.create_key(
                name=args.name,
                scopes=[s.strip() for s in args.scopes.split(",") if s.strip()],
                rpm=args.rpm,
                max_concurrent=args.max_concurrent,
            )
            print(f"key_id: {record.key_id}")
            print(f"token : {token}")
            print("Store the token now; it is only shown once.")
        elif args.action == "list":
            for r in store.list_keys():
                print(f"{r.key_id}  enabled={r.enabled}  scopes={','.join(r.scopes)}  name={r.name}")
        elif args.action in ("disable", "enable"):
            if not args.key_id:
                raise SystemExit("--key-id is required")
            ok = store.set_enabled(args.key_id, args.action == "enable")
            print("updated" if ok else "not found")
    finally:
        store.close()
