# Public auroral data connected to Kristina Lynch

Verified 2026-10-18. **Yes: there are downloadable sounding-rocket measurements and satellite/radar conjunction data directly connected to Lynch's research.** The main entry point is [her Dartmouth publication-data archive](https://rcweb.dartmouth.edu/LynchK/).

**Later application:** [Executable reconstructions and results](../auroral_reconstruction/RESULTS.md) now use these rocket data and six PFISR/DASC events. The application also decodes and verifies the rocket timestamps and source notes. The text below preserves the earlier availability audit.

This follow-up verifies availability and inspects actual files. It does not yet claim a reconstruction result on auroral observations. Raw downloads are ignored by Git; source URLs and hashes are recorded in `sources.json`. Hashes identify the bytes retrieved, rather than a publisher-signed release. The numerical summary is in [results/schema_audit.json](results/schema_audit.json).

## Best starting point: ISINGLASS, March 2, 2017

Lynch was the mission PI; the [lab mission description](https://sites.dartmouth.edu/lynch-rocket-lab/rockets/) ties its multipoint payload development directly to GreenCube. The compact, directly usable input is [flow_data_ig.mat](https://rcweb.dartmouth.edu/LynchK/pubdata/2024_vanirsel_jgr/data/supplementary/flow_data_ig.mat), in the [2024 supplementary archive](https://rcweb.dartmouth.edu/LynchK/pubdata/2024_vanirsel_jgr/data/supplementary/).

I downloaded and decoded its numerical variables: **195 time-ordered samples**, with geographic/geomagnetic east and north flow components, geomagnetic latitude/longitude, and geographic positions labelled as mapped to 100 km. All inspected numeric columns are finite. The source `time` spans about 134–534 seconds. These are processed track data, not untouched instrument telemetry or 195 independent auroral events. The MATLAB datetime and text objects need dedicated decoding before asserting absolute UTC for each row. The inspected numeric columns do not include measurement uncertainties.

The same directory provides `GPSecef.txt`, particle data, `boundaries.mat`, potential maps and `replicate_data_isinglass.mat`. The last is a derived replication product and must not become independent test truth. The boundary file uses MATLAB v7.3 object references; opening the HDF5 container alone does not decode its interpolant objects. Upstream figure scripts document flow units and transformations.

References to cite:

- Clayton, Lynch et al. (2019), [*Two-Dimensional Maps of In Situ Ionospheric Plasma Flow Data Near Auroral Arcs Using Auroral Imagery*](https://doi.org/10.1029/2018JA026440).
- Clayton, Burleigh, Lynch et al. (2021), [*Examining the Auroral Ionosphere in Three Dimensions Using Reconstructed 2D Maps of Auroral Data to Drive the 3D GEMINI Model*](https://doi.org/10.1029/2021JA029749).
- van Irsel, Lynch, Mule & Zettergren (2024), [*Generation of Top-Boundary Conditions for 3D Ionospheric Models Constrained by Auroral Imagery and Plasma Flow Data*](https://doi.org/10.1029/2024JA032722).

The 2024 paper is a **close required comparator**: it already combines arc geometry, plasma-flow tracks and electrostatic reconstruction. Our earlier literature audit omitted it. Its data-availability statement explicitly points to the Dartmouth ISINGLASS inputs and reconstruction tools. This further narrows any novelty claim for geometry-informed constrained reconstruction.

## Stronger validation opportunity: Swarm-over-Poker 2023

Lynch is a co-investigator in [this campaign](https://sites.dartmouth.edu/lynch-rocket-lab/ongoing-projects/) and a coauthor of [van Irsel et al., *Current Closure and Joule Heating in Data-Driven 3-D Auroral Arc Simulations* (2025)](https://doi.org/10.1029/2025JA034311). The [2025 data archive](https://rcweb.dartmouth.edu/LynchK/pubdata/2025_vanirsel_jgr/data/data/) has concrete instrument products:

| Product | Verified contents or availability | Use |
|---|---|---|
| [Swarm](https://rcweb.dartmouth.edu/LynchK/pubdata/2025_vanirsel_jgr/data/data/swarm/) | TII drift CDF/HDF5, errors, electric/magnetic components and quality flags; FAC and Langmuir-probe files also listed | Satellite tracks and component-wise uncertainty |
| [PFISR](https://rcweb.dartmouth.edu/LynchK/pubdata/2025_vanirsel_jgr/data/data/pfisr/) | Five-minute vector products; downloaded Feb 10 file includes velocity, electric field and covariance matrices | Radar comparison with its averaging/resolution respected |
| [DASC](https://rcweb.dartmouth.edu/LynchK/pubdata/2025_vanirsel_jgr/data/data/dasc/) | Imagery directory, inverted energy-flux products and inversion code | Geometry independent of the held-out flow instrument |
| [Event metadata](https://rcweb.dartmouth.edu/LynchK/pubdata/2025_vanirsel_jgr/data/data/events.dat) | Eight spacecraft-event rows, **six distinct conjunction times** | Event/spacecraft grouping; avoid duplicate-event counting |

The six times are February 10 09:51:27 UTC, February 12 10:22:11, March 4 07:30:12 and 10:13:49, March 14 06:49:07, and March 19 08:23:30 (all 2023). February 10 and March 14 each list both Swarm A and C. ESA Swarm is a flown satellite constellation, **not a CubeSat mission built by the Dartmouth lab**.

Actual file audit of the February 10 candidate:

- Swarm A has 89,537 records in the downloaded product; C has 21,871. Both provide 480 samples in a +/-120-second temporal window around the metadata time. This is not yet a spatial conjunction selection.
- The documented quality bits pass 292/480 A horizontal cross-track samples and 0/480 for its other components; none pass for C in this window. This is one quality criterion, not a categorical declaration that every unflagged measurement is unusable. Release notes, calibration flags, error estimates and provider guidance are needed before choosing a defensible benchmark. The 2024 article also discusses caution and instrument-team consultation for along-track flows.
- The PFISR product has 141 time records by 20 magnetic-latitude bins with three velocity components and 3x3 covariances. Its geographic arrays additionally map to 24 altitude levels. Those mapped levels must not be counted as 24 independent instruments or observations. Matching must respect the five-minute averages and the spatial observation operator.

The group's public [Aurora Gemini code](https://github.com/317Lab/aurora_gemini) provides boundary-driver and analysis tools and a Swarm-over-Poker sample setup. It is a more relevant application baseline than generic scattered interpolation alone.

## Simulations, CubeSats and the newest rockets

The archive also contains [2021 GEMINI runs](https://rcweb.dartmouth.edu/LynchK/pubdata/2021_clayton_jgr/data/Simulations/), [2022 ISINGLASS case-study simulations](https://rcweb.dartmouth.edu/LynchK/pubdata/2022_burleigh_jgr/data/Simulations/), and later simulation outputs. These are useful physics-based benchmarks with known simulated fields, but not independently measured ground truth. The [Burleigh et al. 2022 paper](https://doi.org/10.1029/2021JA030242) specifically examines spatial/time limitations.

For **ARCS**, the verified NASA record is an [Auroral Reconstruction CubeSwarm concept study](https://ntrs.nasa.gov/citations/20240008784). I did not verify an orbital ARCS flight-data release; the mission concept must not be cited as if its proposed swarm had produced observations. The lab's [GreenCube page](https://sites.dartmouth.edu/lynch-rocket-lab/greencube/) describes balloon and deployable-payload development.

**GNEISS** is particularly relevant to a future conversation with Lynch: her two rockets [launched February 10, 2026](https://physics.dartmouth.edu/news/2026/03/dartmouth-team-sends-two-rockets-northern-lights), 30 seconds apart, with deployable ion probes and ground observations. I have not verified a public calibrated GNEISS data release. Its successful launch is not evidence that a ready analysis dataset is publicly available.

## Recommended study

First use the 195-sample ISINGLASS input to reproduce the published geometry/electrostatic workflow and establish coordinate, time and uncertainty provenance. Then test whether geometry-aware reconstruction improves prediction of an independently withheld instrument or spacecraft in Swarm-over-Poker, using the published method and a geometry-free model as controls. Hold out the entire instrument/track from fitting, background estimation and tuning; compare through its time/space measurement operator. Exclude replicated maps and model outputs from observational ground truth. Six conjunctions provide a starting point, not enough evidence for unrestricted auroral generalization.

This would connect the new work directly to her measurements while asking a question beyond reproducing a visually plausible interpolation. It remains a proposed evaluation; the current package contains an acquisition/schema/quality audit, not a claimed improvement on these events.

## Reproduce the acquisition audit

From GreenCube:

```bash
.venv/bin/python -m pip install -r research/auroral_data/requirements.txt
.venv/bin/python research/auroral_data/audit.py --fetch
```

The six pinned files total approximately 31 MB. This optional audit is separate from the original benchmark runner and does not alter its 1,484 model evaluation rows. No correspondence has been sent to the investigators.
