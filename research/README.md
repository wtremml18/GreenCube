# Research experiments extending the 2018 Dartmouth thesis

Start with **[RESEARCH_FINDINGS.md](RESEARCH_FINDINGS.md)** for conclusions, numerical evidence, limitations and the publication decision. This is a research package, not a manuscript or a submission.

- [`track_observability`](track_observability): independent Python kernel implementation, dipole audit, track ambiguity, finite-basis observability, arc geometry and public ocean-data experiments.
- [`gaussian_transport`](gaussian_transport): Gaussian position/covariance transport, exact density ambiguity, and sparse-velocity experiments on public ScalarSyn fluid volumes.
- [`auroral_data`](auroral_data): verified public Lynch-group ISINGLASS and Swarm-over-Poker inputs, source references, and an acquisition/schema/quality audit.
- [`auroral_reconstruction`](auroral_reconstruction): executed application to public ISINGLASS flow and six PFISR/DASC events, actual auroral photography, guarded holdouts, image geometry controls, and uncertainty diagnostics. **[Latest real-data conclusions](auroral_reconstruction/RESULTS.md).**
- [Thesis review](track_observability/notes/thesis_review.md), [literature and novelty audit](LITERATURE.md), and [mathematical specification](METHODS.md).

All new work is isolated here. Existing acquisition/calibration code is unchanged. The original DivFree2/DivFree3 MATLAB implementation was not recovered, so these are **new experiments, not a reproduction of the thesis's numerical tables**. Private thesis/email files are excluded. Public raw datasets are downloaded separately and ignored by Git; checksums and extraction instructions are tracked.

## Reproduce

Run from the GreenCube repository root with Python 3.11 or newer (executed here with Python 3.14.2). CPU computation is sufficient; no CUDA or upstream rendering pipeline is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r research/track_observability/requirements.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest research/track_observability/tests research/gaussian_transport/tests -q
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/reproduce.py
```

After running experiments, `research/validate_results.py` checks saved row counts and split separation; `research/generate_manifest.py` refreshes artifact hashes.

The full runner downloads the four small GLAD/LASER CSVs and the 704 MB ScalarSyn archive, verifies their hashes, extracts eight selected volumes (about 228 MB uncompressed NPZ payloads), and regenerates all reported experiments and figures. Allow several minutes, roughly 1 GB of local data storage, and normal scientific Python memory use. Existing downloads are reused. The minimum requirements allow compatible newer packages; `track_observability/requirements-lock.txt` records the complete executed environment, including tools used for source-document extraction.

For individual stages, run the corresponding `scripts/run_*.py` file. In particular:

```bash
.venv/bin/python research/gaussian_transport/scripts/fetch_fluid.py --archive /path/to/scalarsyn.tar.gz
OPENBLAS_NUM_THREADS=1 .venv/bin/python research/gaussian_transport/scripts/run_fluid.py --support-matched
```

The fluid archive may reside outside the repository. `--support-matched` is the principal fluid control; `run_fluid.py` without it intentionally preserves the initial experiment whose lines largely missed the smoke.

## Evidence and conventions

There are 1,484 fitted-model evaluation rows across nine main result CSVs, in addition to analytical and numerical audits. Each row can involve many candidate fits during validation. Synthetic bootstrap intervals resample independent field/noise seeds, not evaluation grid points. They quantify variation under the specified generators, not uncertainty across real-world events. Public ocean subsets overlap and fluid frames belong to one simulation, so no independent-event confidence interval is claimed for those results.

The protocols record exploratory follow-ups and corrections after outcomes were inspected. All specified conditions, including negative results, remain available. `evidence_manifest.json` records SHA-256 hashes for source code and result artifacts at the final verification point. Wall-clock timing fields can differ on reproduction; numerical results should agree within floating-point variation.

The auroral application is a separate optional pipeline (576.6 MB of public inputs). Its [reproduction commands](auroral_reconstruction/README.md#reproduce) run nine additional tests and produce 1,442 individual held-out prediction rows. These are separate from the original 1,484 benchmark evaluation rows. The full `reproduce.py` runner above has not been expanded to download the auroral data automatically.
