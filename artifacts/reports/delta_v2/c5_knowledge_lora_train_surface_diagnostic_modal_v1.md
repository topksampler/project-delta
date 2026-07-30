# delta_v2 c5 LoRA training-surface diagnostic — Modal v1

Status: `implemented`; optimization/surface-fit failure.

This diagnostic replayed the 63 immutable training prompts against the trained
LoRA. It did not add wording, change gold truth, train a model, or authorize
promotion. Each prompt ran twice with byte-identical greedy output.

| frozen training surface | exact |
|---|---:|
| source-card recall | 9/21 |
| verified true assertion | 0/21 |
| deterministic false assertion | 21/21 |

The LoRA answered `no` to every true and false assertion, including the exact
prompts used during training. The c5 failure is therefore not primarily caused
by evaluation paraphrasing. The first candidate did not overcome the base
model's negative-answer prior on its own training surface.

This also limits what may be inferred from the final dev loss of 0.141. That
dev set contains only recall rows; it does not validate balanced boolean
generation. Future training metrics must include surface-stratified generation
readouts or teacher-forced losses.

Evidence:

- run: `delta-v2-c5-knowledge-lora-train-surface-modal-v1`
- launch commit: `9bdb8d7`
- protocol SHA-256:
  `32847e638e91d32b4472fe85f5125d02ad61388804122b6aeccd880e876b9533`
- config SHA-256:
  `1f8a41e312ec156ebc0b3f694c3783e586cb55d03ce5aa59b471be9989ccb567`
- adapter SHA-256:
  `83fb04ae52502432df6b17d2aaa12cfeb12f457df6a77c83f28fed433d9966fa`
- samples SHA-256:
  `957cfd8a5a276875418d5bf45fddcc09472be8d699d8f43744b4e6c2b93a8a39`
- metrics SHA-256:
  `eac1482e1244933caafc3dd70d70c6b2a3f2adbde091340b1bc530a5216e1d48`
- B2 prefix:
  `runs/delta-v2-c5-knowledge-lora-train-surface-modal-v1/`
- model work: 162.44 seconds on NVIDIA A10
- estimated cost: USD 0.0566

The rational next intervention is a fresh base-model LoRA with preregistered
surface weighting toward the existing verified-true rows. It must retain
verified-false rows and keep the frozen evaluation gates unchanged. Adding the
observed evaluation wording to training would contaminate the comparison and
is not authorized.
