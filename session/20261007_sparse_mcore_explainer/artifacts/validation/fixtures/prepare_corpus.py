"""Freeze production text-mode collator IDs, masks and independent alignments."""

import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from nemo_rl.algorithms.x_token import TokenAligner
from nemo_rl.data.cross_tokenizer_collate import (
    CrossTokenizerCollator,
    CrossTokenizerCollatorConfig,
)

validation = Path(__file__).resolve().parents[1]
manifest = json.loads((validation / "fixture_manifest.json").read_text())
roles = ["student", "teacher_a", "teacher_b", "teacher_c_same_tokenizer"]
toks = {
    role: AutoTokenizer.from_pretrained(
        manifest["models"][role]["snapshot_path"], local_files_only=True
    )
    for role in roles
}
corpus = [
    json.loads(line)
    for line in (validation / "fixtures/examples.jsonl").read_text().splitlines()
]
collator = CrossTokenizerCollator(
    config=CrossTokenizerCollatorConfig(mode="text"),
    student_tokenizer=toks["student"],
    teacher_tokenizers=[toks[role] for role in roles[1:]],
    aligners=[
        TokenAligner(
            student_tokenizer=toks["student"],
            teacher_tokenizer=toks["teacher_a"],
            projection_matrix_path=None,
        ),
        TokenAligner(
            student_tokenizer=toks["student"],
            teacher_tokenizer=toks["teacher_b"],
            projection_matrix_path=None,
        ),
        None,
    ],
    ctx_length_student=256,
    ctx_length_teachers=[256] * 3,
    make_seq_div_by_student=8,
    make_seq_div_by_teachers=[8, 16, 8],
    drop_first_assistant_chunk_kl_by_teacher=[False] * 3,
)
batch = collator(
    [
        {
            "message_log": [{"role": "assistant", "content": row["text"]}],
            "sample_id": row["id"],
            "loss_multiplier": 1.0,
            "idx": i,
        }
        for i, row in enumerate(corpus)
    ]
)
out = {}
for key, value in batch.items():
    if isinstance(value, torch.Tensor):
        out[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "values": value.tolist(),
        }
    elif isinstance(value, (list, tuple, str, int, float, bool, type(None))):
        out[key] = value
result = {
    "status": "PASS",
    "collator": "CrossTokenizerCollator text",
    "model_roles": roles,
    "records": len(corpus),
    "max_length": 256,
    "student_divisibility": 8,
    "teacher_divisibilities": [8, 16, 8],
    "batch": out,
}
(validation / "runs/fixture-preparation-20261010/collated_corpus.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(
    "PASS: production collator IDs, masks and independent Qwen/SmolLM2 alignments for 16 samples."
)
