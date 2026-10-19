# Public fluid source

Source: [LagrangianSplats Dataset](https://huggingface.co/datasets/tnx123/LagrangianSplats-Dataset), official release accompanying Tao, Chen and Chu (2026). Inspected implementation commit `180f61a8b56a39f3a71df168c54c87669081acff`; see [their implementation](https://github.com/taoningxiao/LagrangianSplats).

`fetch_fluid.py` downloads `gt/scalarsyn.tar.gz` (703,997,193 bytes), checks upstream SHA-256 `48e1aece0b78e3cb65c9cf26c206971c39564e619d0a142abd98f45986ebf22d`, and copies only density/velocity NPZ files at frames 0,10,20,29. It never executes upstream code and does not extract arbitrary archive paths. Per-file hashes are in `fluid_manifest.json`.

The source arrays are density (100,178,100,1) and velocity (100,178,100,3). Following the upstream ScalarSyn evaluation configuration, we crop to the first (100,150,100) voxels in (z,y,x) array order. Velocity channels are (x,y,z). Coordinates are divided by 100 on every axis, preserving their relative scale. Velocity values retain source units; no dimensional physical time or velocity conversion is asserted.

Only derived results and metadata are tracked. Raw volumes and archives are ignored. The official code repository uses MIT; the dataset card inspected here does not separately state a dataset license, so this package does not redistribute or relicense the raw volumes. The download remains at the authors' source.

This is one simulated smoke sequence, not four independent events. Sparse direct velocity observations are synthetically sampled from the released reference field and perturbed with documented noise. They are not velocities recovered from images. Discrete source divergence is nonzero; no claim of analytically exact incompressibility for every stored voxel value is made.
