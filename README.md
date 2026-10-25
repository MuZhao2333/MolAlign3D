# MolAlign3D

**Fixed-dimensional E(3)-equivariant representations for 3D molecular reconstruction and editing.**

MolAlign3D combines a geometric molecular encoder, pretrained UniMol features,
and a Bayesian Flow Network (BFN) decoder. This repository contains the ZINC
implementation of MolAlign3D and the MolFLAE baseline, including pretrained
model inference, training, reconstruction evaluation, and latent-space editing.

## Papers

**MolAlign3D: Enhancing Fixed-Dimensional E(3)-Equivariant Latent Space for High-Fidelity 3D Molecular Reconstruction and Editing**

Zitao Chen, Jiatong Ji, Yinjun Jia, Wei-Ying Ma, Yanyan Lan

ICML 2026 · [Paper](https://icml.cc/virtual/2026/poster/66574)

**Manipulating 3D Molecules in a Fixed-Dimensional E(3)-Equivariant Latent Space**

Zitao Chen, Yinjun Jia, Zitong Tian, Wei-Ying Ma, Yanyan Lan

NeurIPS 2025 · [Paper](https://arxiv.org/abs/2506.00771)

## Contents

- [Installation](#installation)
- [Pretrained models](#pretrained-models)
- [Inference](#inference)
- [Latent-space experiments](#latent-space-experiments)
- [Data and training](#data-and-training)
- [Evaluation](#evaluation)
- [Repository structure](#repository-structure)
- [Citation](#citation)
- [License and acknowledgments](#license-and-acknowledgments)

## Installation

The tested environment is Linux, Python 3.12, and PyTorch 2.8.0. The package
supports Python 3.11 and 3.12. Run the following commands from the repository root.

```bash
git clone https://github.com/MuZhao2333/MolAlign3D.git
cd MolAlign3D
```

For a CUDA 12.8 environment:

```bash
bash scripts/install.sh --backend cuda128
source .venv/bin/activate
```

For CPU inference:

```bash
bash scripts/install.sh --backend cpu
source .venv/bin/activate
```

The installer creates `.venv`, installs matching PyTorch Geometric extensions
and the dependencies in `requirements.txt`, and installs the project in editable
mode. Use `--backend existing` with an existing PyTorch 2.8.0 CPU or CUDA 12.8
environment.

## Pretrained models

| Model | Latent nodes | Structural Zh per node | Decoder features per node | Training epochs | Decode steps |
|---|---:|---:|---:|---:|---:|
| MolAlign3D | 8 | 16 | 80 = 16 structural + 64 UniMol | 15 | 100 |
| MolFLAE | 10 | 32 | 32 | 25 | 100 |

Each latent node also has a three-dimensional coordinate Zx. MolAlign3D splits
the frozen 512-dimensional UniMol representation into eight 64-dimensional
blocks and concatenates each block with its corresponding structural feature.

Download and extract the pretrained weights from GitHub Release v1.3.0
using the GitHub CLI:

```bash
gh release download v1.3.0 --repo MuZhao2333/MolAlign3D \
  --pattern '*-weights.tar.gz' --dir outputs/downloads
tar -xzf outputs/downloads/molalign3d-zinc-weights.tar.gz
tar -xzf outputs/downloads/molflae-zinc-weights.tar.gz
```

The archives extract to the following paths:

| Path | Asset |
|---|---|
| `weights/molalign3d/model.ckpt` | MolAlign3D ZINC checkpoint, epoch 14, optimizer step 136514 |
| `weights/molalign3d/mol_pre_no_h_220816.pt` | Pretrained UniMol encoder |
| `weights/molalign3d/mol.dict.txt` | UniMol atom dictionary |
| `weights/molflae/model.ckpt` | MolFLAE ZINC checkpoint, epoch 24 |

Import the checkpoint assets with SHA256 verification:

```bash
python scripts/import_checkpoints.py \
  --molalign3d /path/to/molalign3d.ckpt \
  --molflae /path/to/molflae.ckpt \
  --unimol /path/to/mol_pre_no_h_220816.pt \
  --dictionary /path/to/mol.dict.txt
```

Checkpoint identities and checksums are recorded in
[`src/molalign3d/configs/checkpoint.json`](src/molalign3d/configs/checkpoint.json)
and [`src/molflae/configs/checkpoint.json`](src/molflae/configs/checkpoint.json).
Set `MOLALIGN3D_WEIGHTS_DIR` or `MOLFLAE_WEIGHTS_DIR` to use another asset
directory; each directory must contain the filenames listed above for that model.

## Inference

Reconstruct a molecule with either model:

```bash
molalign3d --smiles 'CC(=O)Oc1ccccc1C(=O)O' \
  --device auto --steps 100 --output outputs/molalign3d/reconstruction.json

molflae --smiles 'CC(=O)Oc1ccccc1C(=O)O' \
  --device auto --steps 100 --output outputs/molflae/reconstruction.json
```

`--device auto` selects an available GPU. The default device is CPU. To encode
an existing 3D conformer:

```bash
molalign3d --experiment encode --sdf /path/to/molecule.sdf \
  --device auto --output outputs/molalign3d/encoding.json
```

Both commands accept SMILES or a single 3D SDF/MOL record. SMILES inputs are
converted to 3D conformers using RDKit ETKDGv3 and UFF optimization when
parameters are available. Supported inputs contain 2–80 heavy atoms drawn from
C, N, O, F, P, S, Cl, Br, and I.

The JSON output contains molecular coordinates, latent features, checkpoint
metadata, and experiment results. Decoding also writes an SDF file beside the
JSON when a molecular structure can be constructed. Outputs retain chemical
validity and fragment counts. An encoding operation returns latent features
without running the decoder.

The per-molecule commands support `encode`, `reconstruct`, `atom-edit`,
`interpolate`, `swap`, and `wander`. Run `molalign3d --help` or `molflae --help`
for their options.

## Latent-space experiments

The batch experiment runner supports reconstruction, atom-number editing,
interpolation, and Zx/Zh exchange with either model. Use original 3D conformers
for comparisons with the published editing experiments.

### Atom-number editing

Provide a single SDF file or a directory of SDF files. Each molecule is encoded
once; decoding varies its heavy-atom count while retaining its latent code.

```bash
python scripts/latent_experiments.py \
  --model molalign3d --molecules /path/to/molecules \
  --experiments atom-edit --radius 2 --steps 100 --device auto \
  --output outputs/atom-edit
```

The default radius evaluates atom-count changes of −2, −1, 0, +1, and +2.

### Interpolation and feature exchange

Provide a directory containing endpoint pairs named `1_start.sdf`, `1_end.sdf`,
`2_start.sdf`, `2_end.sdf`, and so on:

```bash
python scripts/latent_experiments.py \
  --model molalign3d --pairs /path/to/pairs \
  --experiments interpolation swap --frames 12 --steps 100 --device auto \
  --output outputs/interpolation-swap
```

Pair definitions can also be supplied as a JSON file. SDF paths are relative
to that file; `record` selects a zero-based record in a multi-record SDF:

```json
[
  {
    "name": "pair-1",
    "a": {"sdf": "start.sdf", "record": 0},
    "b": {"sdf": "end.sdf", "record": 0}
  }
]
```

Set `--model molflae` to run the same experiments with the baseline.
For input-anchored exploration, the runner also supports `wander` with
`--space zx`, `--space zh`, or `--space both`, and `--step-size` controlling
the perturbation scale.

### Protocol and outputs

The batch runner uses seed 42 and 100 decoding steps by default. Atom editing
encodes the complete input collection together. Pair experiments encode each
endpoint independently in its centered coordinate frame. Interpolation mixes
Zx and the complete decoder feature condition; both exchange directions use
endpoint A's atom count. Sampling uses a continuous random-number stream
within each experiment.

Default decoding batches contain 500 atom-edit frames, 50 interpolation pairs,
or all exchange pairs. Batch composition affects MolAlign3D's UniMol feature
normalization. Preserve input order, batch sizes, conformers, and the software
environment when comparing runs. The per-molecule commands expose additional
interactive settings; use the batch runner for the protocol described here.

Each run requires a new or empty output directory and writes:

| File | Contents |
|---|---|
| `inputs.json` | Input records, pair definitions, and input file hashes |
| `run.json` | Arguments, software versions, checkpoint identity, and protocol |
| `<experiment>/latents.pt` | Full-precision structural Zh, Zx, and complete decoder conditions |
| `<experiment>/references.sdf` | Reference structures with input identifiers |
| `<experiment>/records.jsonl` | Frame IDs, generated atoms and coordinates, validity, and metrics |
| `<experiment>/predictions.sdf` | Generated molecular structures with frame identifiers |
| `<experiment>/summary.json` | Experiment statistics and metric counts |
| `summary.json` | Run-level experiment statistics |

Metrics include MACCS Tanimoto, in-situ Shape Tanimoto, atom-edit MCS-IoU,
and interpolation descriptor Pearson correlations. Similarity statistics use
sanitized pairs and report their sample counts; chemical validity uses all
planned frames. MCS searches have a two-second timeout, with incomplete
searches reported separately as lower bounds. Undefined correlations are
recorded as null. Frame identifiers remain stable when construction fails.

## Data and training

Training uses the ZINC multi-conformer database `ZINC_9M_10confs.lmdb` and a
directory containing `train_indices.pt`, `val_indices.pt`, and
`test_indices.pt`. Supply the database separately. Download the archived
index files with:

```bash
gh release download v1.3.0 --repo MuZhao2333/MolAlign3D \
  --pattern zinc-original-splits.tar.gz --dir outputs/downloads
tar -xzf outputs/downloads/zinc-original-splits.tar.gz
```

The archive extracts to `data/original-splits/`. Index values refer to the
LMDB cursor's lexicographic byte-key order.

The archived split contains 9,318,809 training records, 931 validation records,
and 95 test records. Counts and checksums are recorded in
[`data/splits-manifest.json`](data/splits-manifest.json). Hydrogen atoms are
removed. MolAlign3D samples two conformers per molecule during training;
MolFLAE samples one.

Train either model using its configuration:

```bash
bash scripts/train_zinc.sh molalign3d \
  /path/to/ZINC_9M_10confs.lmdb /path/to/splits outputs/train-molalign3d

bash scripts/train_zinc.sh molflae \
  /path/to/ZINC_9M_10confs.lmdb /path/to/splits outputs/train-molflae
```

| Setting | MolAlign3D | MolFLAE |
|---|---:|---:|
| Epochs | 15 | 25 |
| Molecules per GPU | 64 | 200 |
| Gradient accumulation | 2 | 1 |
| GPUs in default configuration | 8 | 8 |
| Adam learning rate | 0.0005 | 0.0005 |
| Adam β₁ / β₂ | 0.95 / 0.999 | 0.95 / 0.999 |

Append `--devices 1` to select a single GPU, or use a comma-separated GPU list
such as `--devices 0,1`. The effective batch size is molecules per GPU × GPU
count × gradient accumulation. Set `--batch-size`, `--epochs`, or `--lr` to
override the corresponding configuration field.

Inspect the resolved configuration before training:

```bash
python scripts/train.py --model molalign3d \
  --data /path/to/ZINC_9M_10confs.lmdb --splits /path/to/splits \
  --accelerator gpu --output outputs/config-preview --dry-run
```

Training writes `resolved-config.yaml`, run metadata, CSV logs, and resumable
checkpoints under the selected output directory. Use `--resume` with a saved
checkpoint to restore training state, or `--init-checkpoint` to initialize a
new training run from model weights.

Both models have been checked with two-GPU NCCL DDP training on real ZINC
conformers, including four optimizer updates, 100-step validation decoding,
and checkpoint saving and loading.

## Evaluation

Export a subset with its existing 3D conformers:

```bash
python scripts/export_zinc_subset.py \
  --lmdb /path/to/ZINC_9M_10confs.lmdb --indices-dir /path/to/splits \
  --per-split 100 --output outputs/zinc-subset
```

Evaluate reconstruction on the exported test split:

```bash
python scripts/evaluate.py --model molalign3d \
  --data outputs/zinc-subset/molecules.jsonl \
  --splits outputs/zinc-subset/splits.json --split test \
  --steps 100 --device auto --output outputs/evaluation-molalign3d
```

Use `--model molflae` for the baseline. The evaluator writes per-record
predictions, an SDF collection, and `metrics.json` with completion rate,
connected validity, graph agreement, fingerprint similarity, and graph-matched
RMSD. Exported reference bonds are inferred from coordinates; the export
metadata records this convention.

The shared command `molalign3d-research` also provides single-target molecular
property probing and encoder fine-tuning from CSV data. Run
`molalign3d-research property-probe --help` or
`molalign3d-research finetune --help` for the required fields and options.

## Citation

Please cite MolAlign3D when using the model, and MolFLAE when using the baseline:

```bibtex
@inproceedings{chen2026molalign3d,
  title={MolAlign3D: Enhancing Fixed-Dimensional {E(3)}-Equivariant Latent Space
         for High-Fidelity {3D} Molecular Reconstruction and Editing},
  author={Chen, Zitao and Ji, Jiatong and Jia, Yinjun and Ma, Wei-Ying and Lan, Yanyan},
  booktitle={International Conference on Machine Learning},
  year={2026}
}

@inproceedings{chen2025molflae,
  title={Manipulating {3D} Molecules in a Fixed-Dimensional {E(3)}-Equivariant Latent Space},
  author={Chen, Zitao and Jia, Yinjun and Tian, Zitong and Ma, Wei-Ying and Lan, Yanyan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

Machine-readable citation metadata is available in [`CITATION.cff`](CITATION.cff).

## License and acknowledgments

The project is distributed under the [MIT license](LICENSE). The included
UniMol implementation retains its
[MIT license](src/molalign3d/vendor/unimol/LICENSE), copyright DP Technology.
See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution.
