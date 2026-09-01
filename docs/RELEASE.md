# Release procedure

1. Start from a clean clone and install `requirements-dev.txt`.
2. Run `make test` and `make audit`.
3. Rebuild the submitted release with `make rebuild-submitted`; do not use the
   methodological-retraining package target for the historical artifact.
4. Run one four-case smoke test and a full evaluation-sized rehearsal on the
   target GPU/container.
5. Confirm every input ID occurs exactly once and every result matches the wire
   grammar.
6. Upload the exact verified adapter or submitted ZIP as the release asset. Do
   not commit large artifacts as ordinary Git blobs.
7. Publish SHA-256, byte size, tested base-model revision, dependency lock and
   measured wall time.
8. Scan the clean repository history and archive contents for secrets and
   machine-specific paths.
9. Confirm publication rights for competition-derived code, data and adapter.
