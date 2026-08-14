# Forced Alignment Engine

Stateless Railway adapter for Media Creator's mandatory ElevenLabs forced-alignment gate. It downloads the signed source once, creates sample-accurate bounded FLAC windows, aligns them, validates monotonic word lineage, and returns one verified timeline.

## Required environment

- `ALIGNMENT_SHARED_SECRET`: high-entropy secret shared only with the BullMQ worker.
- `ELEVENLABS_API_KEY`: provider key already used by Media Creator.
- `ALIGNMENT_MAX_CONCURRENCY=1`: one bounded alignment per service replica.
- `ALIGNMENT_CHUNK_SECONDS=240`: stays below the provider's long-form/file limits.

The service exposes `GET /health` and authenticated `POST /align`. It is stateless and needs no volume.
