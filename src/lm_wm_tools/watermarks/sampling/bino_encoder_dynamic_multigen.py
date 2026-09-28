from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import uuid

import torch

from .bino_encoder_dynamic import (
    BinoEncoderState,
    DynamicBinoEncoder,
    normalize_remaining_bits,
)


def _clone_state(state: BinoEncoderState, *, device: torch.device) -> BinoEncoderState:
    payload = state.payload.to(device=device, dtype=torch.int32).clone()
    cloned = BinoEncoderState(
        state.payload_size,
        payload,
        wait=state.wait,
        remaining_bits=state.remaining_bits,
        method=state.method,
    )
    cloned.payload_matches = state.payload_matches.to(
        device=device, dtype=torch.int32
    ).clone()
    cloned.running_counts = state.running_counts.to(
        device=device, dtype=torch.int32
    ).clone()
    cloned.n_trials = int(state.n_trials)
    return cloned


def _state_to_dict(state: BinoEncoderState, *, segment_id: str | None) -> dict[str, object]:
    return {
        "payload_size": int(state.payload_size),
        "payload": state.payload.detach().cpu().to(dtype=torch.int32).tolist(),
        "payload_matches": state.payload_matches.detach()
        .cpu()
        .to(dtype=torch.int32)
        .tolist(),
        "running_counts": state.running_counts.detach()
        .cpu()
        .to(dtype=torch.int32)
        .tolist(),
        "n_trials": int(state.n_trials),
        "wait": int(state.wait),
        "remaining_bits": [int(bit) for bit in state.remaining_bits],
        "method": state.method,
        "last_segment_id": segment_id,
    }


def _state_from_dict(data: dict[str, object], *, device: torch.device) -> BinoEncoderState:
    payload = torch.tensor(data["payload"], device=device, dtype=torch.int32)
    state = BinoEncoderState(
        int(data["payload_size"]),
        payload,
        wait=int(data["wait"]),
        remaining_bits=normalize_remaining_bits(data["remaining_bits"]),
        method=str(data["method"]),
    )
    state.payload_matches = torch.tensor(
        data["payload_matches"],
        device=device,
        dtype=torch.int32,
    )
    state.running_counts = torch.tensor(
        data["running_counts"],
        device=device,
        dtype=torch.int32,
    )
    state.n_trials = int(data["n_trials"])
    return state


class _BinoEncoderStateStore:
    def __init__(self, *, state_dir: str, state_id: str):
        state_dir_path = Path(state_dir)
        state_dir_path.mkdir(parents=True, exist_ok=True)

        state_key = str(state_id).strip()
        if not state_key:
            raise ValueError("multigen_state_id must be a non-empty string.")

        digest = hashlib.sha256(state_key.encode("utf-8")).hexdigest()
        self.state_path = state_dir_path / f"{digest}.json"
        self.lock_path = state_dir_path / f"{digest}.lock"

    @contextmanager
    def locked(self):
        with open(self.lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def load_unlocked(self, fallback_state: BinoEncoderState) -> BinoEncoderState:
        if not self.state_path.exists():
            state = _clone_state(fallback_state, device=fallback_state.payload.device)
            self.save_unlocked(state, segment_id=None)
            return state

        with open(self.state_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return _state_from_dict(data, device=fallback_state.payload.device)

    def save_unlocked(self, state: BinoEncoderState, *, segment_id: str | None) -> None:
        payload = _state_to_dict(state, segment_id=segment_id)
        temp_path = self.state_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(temp_path, self.state_path)


class DynamicBinoEncoderMultiGen(DynamicBinoEncoder):
    def __init__(
        self,
        *args,
        multigen_state_dir: str | None = None,
        multigen_state_id: str | None = None,
        multigen_segment_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._state_sync_suspended = False
        self.multigen_segment_id = (
            str(multigen_segment_id)
            if multigen_segment_id is not None
            else str(uuid.uuid4())
        )

        if multigen_state_dir is None and multigen_state_id is None:
            self._state_store: _BinoEncoderStateStore | None = None
            return

        if not multigen_state_dir or not multigen_state_id:
            raise ValueError(
                "multigen_state_dir and multigen_state_id must both be provided "
                "when using DynamicBinoEncoderMultiGen."
            )

        self._state_store = _BinoEncoderStateStore(
            state_dir=multigen_state_dir,
            state_id=multigen_state_id,
        )
        self._reload_state()

    def _reload_state(self) -> None:
        if self._state_store is None:
            return
        with self._state_store.locked():
            self.state = self._state_store.load_unlocked(self.state)

    def sample(
        self,
        seed: int,
        context_hashes: torch.Tensor | int,
        tokens: torch.Tensor,
        embed_message: bool = True,
        depth: int = 1,
    ) -> torch.Tensor:
        if self._state_store is None or self._state_sync_suspended:
            return super().sample(
                seed=seed,
                context_hashes=context_hashes,
                tokens=tokens,
                embed_message=embed_message,
                depth=depth,
            )

        with self._state_store.locked():
            self.state = self._state_store.load_unlocked(self.state)
            return super().sample(
                seed=seed,
                context_hashes=context_hashes,
                tokens=tokens,
                embed_message=embed_message,
                depth=depth,
            )

    def update_state(
        self,
        tokens: list[int] | torch.Tensor | None = None,
        seeding_scheme=None,
        rng_device: torch.device | None = None,
        context_size: int | None = None,
        seed: int | None = None,
        context_hash: int | None = None,
        token: int | None = None,
    ):
        if self._state_store is None:
            return super().update_state(
                tokens=tokens,
                seeding_scheme=seeding_scheme,
                rng_device=rng_device,
                context_size=context_size,
                seed=seed,
                context_hash=context_hash,
                token=token,
            )

        with self._state_store.locked():
            self.state = self._state_store.load_unlocked(self.state)
            self._state_sync_suspended = True
            try:
                result = super().update_state(
                    tokens=tokens,
                    seeding_scheme=seeding_scheme,
                    rng_device=rng_device,
                    context_size=context_size,
                    seed=seed,
                    context_hash=context_hash,
                    token=token,
                )
            finally:
                self._state_sync_suspended = False

            self._state_store.save_unlocked(
                self.state,
                segment_id=self.multigen_segment_id,
            )
            return result
