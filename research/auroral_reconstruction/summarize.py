"""Check stored evidence and regenerate the conclusions from final result tables."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent


def main():
    p=pd.read_csv(ROOT/'results/predictions.csv')
    metrics=pd.read_csv(ROOT/'results/event_metrics.csv')
    sensitivity=pd.read_csv(ROOT/'results/sensitivity.csv')
    meta=json.loads((ROOT/'results/data_metadata.json').read_text())
    selections=json.loads((ROOT/'results/selections.json').read_text())
    assert len(p)==1442 and len(metrics)==49
    assert not p.duplicated(['event','model','row']).any()
    assert np.isfinite(p[['observed_e','observed_n','predicted_e','predicted_n','squared_error']]).all().all()
    np.testing.assert_allclose(p.squared_error,(p.predicted_e-p.observed_e)**2+(p.predicted_n-p.observed_n)**2,rtol=1e-12)
    for s in selections:
        tr=np.array(s['train_rows']);te=np.array(s['test_rows']);guard=2 if s['event']=='isinglass' else 1
        assert np.min(abs(tr[:,None]-te[None,:]))>guard
    for (event,name),g in p.groupby(['event','model']):
        row=metrics[(metrics.event==event)&(metrics.model==name)].iloc[0]
        np.testing.assert_allclose(row.rmse,np.sqrt(g.squared_error.mean()),rtol=1e-12)
    for path in (ROOT/'results').glob('*_field_map.npz'):
        with np.load(path) as a:
            for k in a.files:
                assert np.isfinite(a[k]).all()
                if k.endswith('covariance'):
                    assert np.linalg.eigvalsh(a[k]).min()>-1e-6
    r=metrics[metrics.event!='isinglass'];t=r.pivot(index='event',columns='model',values='rmse')
    rocket=metrics[metrics.event=='isinglass'].set_index('model')
    rmse=t.mean();mse=r.groupby('model').mse.mean()
    source=json.loads((ROOT/'sources.json').read_text())
    verification=dict(primary_prediction_rows=len(p),event_model_summaries=len(metrics),radar_events=6,radar_bins=76,rocket_samples=130,
                      numerical_unit_tests=9,source_files=len(source),source_bytes=sum(i['bytes'] for i in source),
                      checks=['unique held-out observation per event/model','outer train/test separation with guard','all physical predictions finite',
                              'loss independently recomputed from vector components','reported RMSE agrees with saved predictions',
                              'exported latent covariance positive semidefinite'],
                      experiment_exit_code=0,scope='Processed radar-bin and rocket-segment holdout; no independent-track validation')
    (ROOT/'results/verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    lines=['# Application to public auroral measurements and photography','',
           '**Executed 2026-10-18. Conclusions before any paper drafting.**','',
           'The public data application works technically: actual auroral camera products now inform physical field models, and measured observations are withheld from fitting and tuning. The results do **not** establish that image geometry reliably improves reconstruction. The simpler unconstrained model wins on the six-event radar aggregate; the physical model helps the full rocket track, while its image geometry adds no improvement.','',
           '## What is connected','',
           '- ISINGLASS processed sounding-rocket flow and the corresponding 2017 optical energy-flux inversion: 130 of 195 measurements fall inside the optical map.',
           '- Six Swarm-over-Poker campaign conjunctions: 76 retained PFISR latitude-bin electric-field estimates, matched to DASC calibrated 428 nm imagery and its 110 km geographic registration. This benchmark does not use satellite drift as test truth.',
           '- Three actual camera PNG exposures at 428, 558 and 630 nm, displayed separately. No generated auroral imagery is used.',
           '- 17 publicly downloadable files, 576.6 MB, pinned by SHA-256. [Provenance, methods and references](README.md).','',
           '![Actual all-sky exposures](figures/actual_auroral_photographs.png)','',
           '## Rocket result: physics helps this full track, geometry does not','',
           '| Model | Held-out vector RMSE (m/s) | Conditional 95% ellipse coverage |','|---|---:|---:|']
    for name,label in [('constant','Constant'),('independent','Unconstrained'),('electrostatic','Divergence-free, isotropic'),('straight_anisotropic','Divergence-free, straight anisotropy'),('image_aligned','Divergence-free, image curvature')]:
        z=rocket.loc[name];c='—' if pd.isna(z.coverage95) else f'{100*z.coverage95:.1f}%'
        lines.append(f'| {label} | {z.rmse:.1f} | {c} |')
    benefit=100*(1-rocket.loc['electrostatic','rmse']/rocket.loc['independent','rmse'])
    penalty=100*(rocket.loc['image_aligned','rmse']/rocket.loc['electrostatic','rmse']-1)
    lines += ['',f'The physical kernel reduces RMSE by **{benefit:.1f}%** relative to the unconstrained kernel on guarded contiguous holdouts. The optical version raises RMSE by **{penalty:.1f}%** relative to the isotropic physical model. These are same-track predictions, not proof of accuracy between tracks.',
              '',f"The image-model 95% ellipses cover only **{100*rocket.loc['image_aligned','coverage95']:.1f}%** of held-out vectors under the provisional error model. Its uncertainty is strongly undercalibrated. A physically constrained covariance does not automatically deliver trustworthy uncertainty.",'',
              '![Rocket reconstructions](figures/isinglass_maps.png)','',
              '## Radar result: mixed event effects, no aggregate advantage','',
              '| Event (UTC date / crossing) | Unconstrained | Physical isotropic | Image curved |','|---|---:|---:|---:|']
    for event,row in t.iterrows():lines.append(f'| {event} | {row.independent:.2f} | {row.electrostatic:.2f} | {row.image_aligned:.2f} |')
    lines += [f'| Equal-event mean RMSE | **{rmse.independent:.2f}** | {rmse.electrostatic:.2f} | {rmse.image_aligned:.2f} |','',
              'Units are mV/m. The mean is the arithmetic mean of the six event RMSE values, not the square root of pooled MSE.',
              '',f"Image geometry improves on the unconstrained model in **{int((t.image_aligned<t.independent).sum())}/6 events**, but its equal-event mean RMSE is **{100*(rmse.image_aligned/rmse.independent-1):.1f}% worse**. Its event-mean MSE is also worse ({mse.image_aligned:.1f} versus {mse.independent:.1f} (mV/m)²). Against the isotropic physical model it reduces mean RMSE by {100*(1-rmse.image_aligned/rmse.electrostatic):.1f}%, which does not make it the best comparator.",'',
              '![All event outcomes](figures/radar_comparison.png)','',
              '## Why the optical link is not yet a reliable predictor','',
              '![All optical geometry fits](figures/six_event_geometry.png)','',
              'The image-only quadratic ridge works visually on several coherent arcs. It is poor for the two March 4 examples: disconnected/truncated bright structures and competing arcs are forced into one curve. Optical ridge residuals are about 70 and 40 km. This is an exposed failure of the simple extraction model, not evidence that photography cannot help. Other events can still fail even with a visually good ridge.',
              '', 'The fixed 4:1 correlation-length ratio and assumed relation between a brightness ridge and flow structure are additional model assumptions. The straight, linear-image and reversed-image controls are retained in the complete table. This study does not exhaust better segmentations or uncertainty-aware geometry models.', '',
              '## Time and quality sensitivity','',
              'Decoding the MATLAB datetime payload resolved the rocket timing and recovered a source warning about its approximate geomagnetic vector rotation. The main optical snapshot is 07:54:10 UTC; the full selected track spans several minutes. After this audit, a post-hoc test restricted observations to +/-30 seconds of four optical snapshots:', '',
              '| Optical snapshot UTC | Samples | Unconstrained | Physical isotropic | Image curved |','|---|---:|---:|---:|---:|']
    times={0:'07:52:40',45:'07:53:25',90:'07:54:10',135:'07:54:55'}
    matched=sensitivity[sensitivity.study=='rocket_matched_30s']
    for setting,g in matched.groupby('setting'):
        z=g.set_index('model');lines.append(f"| {times[int(setting)]} | {int(z.iloc[0]['n'])} | {z.loc['independent','rmse']:.1f} | {z.loc['electrostatic','rmse']:.1f} | {z.loc['image_aligned','rmse']:.1f} |")
    lines += ['', 'Units are m/s. These short, overlapping windows yield mixed outcomes. They reinforce the absence of a robust image-geometry benefit and are not four independent events. Rocket noise assumptions of 50 and 200 m/s, other full-track image snapshots, and radar error thresholds of 20 and 50 mV/m are retained in [sensitivity.csv](results/sensitivity.csv).', '',
              'The radar quality threshold materially affects some absolute errors. For example, reducing the threshold removes a large February 10 excursion. Reported gains must therefore be read with the retained-bin lists and sensitivity results, not just the headline average.', '',
              '## What can and cannot be concluded','',
              '**Established here:** public Lynch-group data can be ingested reproducibly; camera geometry can enter a physical kernel without replicating measurements; guarded holdouts expose failures that smooth maps conceal; conditional uncertainty can be quantitatively overconfident.', '',
              '**Not established:** a new interpolation formula, a reliable image-driven improvement, calibrated two-dimensional uncertainty, independent-track accuracy, or a reproduction outperforming the published 2024 reconstruction method.', '',
              'The next worthwhile experiment is a multi-arc geometry model with uncertainty, tested against a separately withheld spacecraft/rocket track or independent instrument through its actual averaging operator. The present radar products average roughly five minutes and are already inversion outputs; their bins are not independent raw measurements. The existing Swarm quality/mapping issue has not been bypassed.', '',
              '## Evidence and reproduction','',
              '- [Every primary held-out prediction](results/predictions.csv), [event metrics](results/event_metrics.csv), [outer split membership and selected hyperparameters](results/selections.json), [data/timing/geometry metadata](results/data_metadata.json).',
              '- [Rocket uncertainty and model disagreement](figures/isinglass_uncertainty.png), [radar uncertainty and model disagreement](figures/20230210_uncertainty.png). Exported `*_field_map.npz` files contain coordinates, two model means, conditional latent covariance matrices, and distance to the nearest measurement.',
              f'- {len(p):,} primary held-out prediction rows, 49 event/model summaries, 9 passing numerical/data tests, 17 verified source hashes. [Integrity checks](results/verification.json).',
              '- [Methods and commands](README.md). Run `fetch.py`, the unit tests, `run.py --sensitivity`, and `summarize.py`. No manuscript has been drafted.','']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines))
    print('Checked',len(p),'predictions; wrote RESULTS.md and verification.json.')

if __name__=='__main__':main()
