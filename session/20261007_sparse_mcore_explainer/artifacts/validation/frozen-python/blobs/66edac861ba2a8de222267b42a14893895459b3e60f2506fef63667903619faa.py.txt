"""Bounded CUDA/CPU common-K tie probe on frozen R3 teacher observations."""

import argparse
import hashlib
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    known = torch.load(
        args.run / "controller-reference/items0-1-rank0.pt",
        map_location="cpu",
        weights_only=False,
    )
    # Plaintext corpus masks all repeated EOS padding. Verify the exact shifted
    # validity definition against the surviving actual DP0 capture before using
    # the same collator definition for DP1, whose failed capture was not saved.
    eos_id = 128001
    known_mask = known["data"]["kd_token_mask"][:, 1:].bool()
    assert torch.equal(known_mask, known["data"]["input_ids"][:, 1:] != eos_id)
    cases = []
    for item_ids in ((0, 1), (4, 5)):
        records = [
            torch.load(
                args.run / f"teacher-observations/teacher1/{i}.pt",
                map_location="cpu",
                weights_only=False,
            )
            for i in item_ids
        ]
        logits = torch.stack([r["logits"] for r in records])
        ids = torch.stack([r["input_ids"] for r in records])
        mask = ids != eos_id
        valid = torch.cat((mask[:, 1:], torch.zeros_like(mask[:, :1])), dim=1)
        importance = (
            logits.masked_fill(~valid[..., None], -torch.inf)
            .flatten(0, 1)
            .max(0)
            .values
        )
        k = int(known["loss_config"]["vocab_topk"])
        cpu = importance.topk(k).indices.sort().values
        cuda = importance.cuda().topk(k).indices.sort().values.cpu()
        cutoff = importance[cpu].min()
        ties = (importance == cutoff).nonzero().flatten()
        assert importance[cuda].min() == cutoff
        assert set((importance > cutoff).nonzero().flatten().tolist()).issubset(
            cuda.tolist()
        )
        cases.append(
            {
                "item_ids": list(item_ids),
                "valid_predictors": valid.sum(-1).tolist(),
                "k": k,
                "cutoff": float(cutoff),
                "strictly_above": int((importance > cutoff).sum()),
                "cutoff_tied_ids": ties.tolist(),
                "cpu_support": cpu.tolist(),
                "cuda_support": cuda.tolist(),
                "cpu_only": sorted(set(cpu.tolist()) - set(cuda.tolist())),
                "cuda_only": sorted(set(cuda.tolist()) - set(cpu.tolist())),
            }
        )
    report = {
        "status": "PASS_PROBE",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "cases": cases,
        "scope": "Discrete topk support only on saved real teacher logits; no model/loss/gradient acceptance",
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
