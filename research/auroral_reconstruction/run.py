"""Reproduce nested holdouts and publication-quality diagnostic figures."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from data import ROOT, EVENTS, radar_event, rocket_event, DATA, electric_output_scale
from model import contiguous_splits, fit_tuned

MODELS = ['constant','independent','electrostatic','straight_anisotropic','image_linear','image_aligned','image_reversed']
LABELS = dict(constant='Constant', independent='Unconstrained', electrostatic='Physical / isotropic',
              straight_anisotropic='Physical / straight', image_linear='Image / linear', image_aligned='Image / curved', image_reversed='Image / reversed')
COLORS = dict(zip(MODELS, ['#888888','#C45735','#2870A2','#A38731','#7858A6','#13836B','#CF73A1']))


def evaluate(event, names=MODELS):
    x, y, noise, ids = (event[k] for k in ['x','y','noise','ids'])
    scale, physical_y = event['output_scale'], event['physical_y']
    guard = 2 if event['flow'] else 1
    output, selections = [], []
    for fold, (train, test) in enumerate(contiguous_splits(ids, 4, guard)):
        assert not set(ids[train]) & set(ids[test])
        assert np.min(abs(ids[train, None] - ids[test][None, :])) > guard
        for name in names:
            if name == 'constant':
                pred = np.tile(physical_y[train].mean(0), (len(test),1))
                variance = None
                best = {}
            else:
                model, best = fit_tuned(x[train], y[train], noise[train], ids[train], name, event['polynomial'], event['flow'], scale[train])
                pred, variance = model.predict(x[test], True)
                variance += noise[test] + np.eye(2)[None] * best['floor'] ** 2
                pred *= scale[test]
                variance *= scale[test,:,None] * scale[test,None,:]
            selections.append(dict(event=event['name'],model=name,fold=fold,train_rows=ids[train].tolist(),test_rows=ids[test].tolist(),**best))
            for j, index in enumerate(test):
                error = pred[j] - physical_y[index]
                mahal = float(error @ np.linalg.solve(variance[j], error)) if variance is not None else np.nan
                denom = np.linalg.norm(pred[j])*np.linalg.norm(physical_y[index])
                angle = float(np.rad2deg(np.arccos(np.clip(pred[j]@physical_y[index]/denom,-1,1)))) if denom>1e-12 else np.nan
                output.append(dict(event=event['name'],model=name,fold=fold,row=int(ids[index]),
                                   x_km=x[index,0],y_km=x[index,1],observed_e=physical_y[index,0],observed_n=physical_y[index,1],
                                   predicted_e=pred[j,0],predicted_n=pred[j,1],squared_error=float(error@error),
                                   angle_degrees=angle,mahalanobis2=mahal,covered95=float(mahal <= 5.991464547) if np.isfinite(mahal) else np.nan))
    return pd.DataFrame(output), selections


def image_panel(ax, event, title=None):
    xy, im = event['image_xy'], event['image']
    if event['flow']:
        xl, yl = (-140,140),(-78,78)
    else:
        xl, yl = (-180,180),(-80,420)
    mask = (xy[...,0]>xl[0])&(xy[...,0]<xl[1])&(xy[...,1]>yl[0])&(xy[...,1]<yl[1])&np.isfinite(im)
    vmin, vmax = np.nanquantile(im[mask], [.1,.99])
    ax.pcolormesh(xy[...,0],xy[...,1],im,cmap='gray',shading='auto',vmin=vmin,vmax=vmax,rasterized=True)
    xx=np.linspace(*xl,200);p=np.polynomial.Polynomial(event['polynomial'])
    ax.plot(xx,p(xx),color='#E9B933',lw=1.7,label='Optical ridge')
    ax.set(xlim=xl,ylim=yl,xlabel=('Magnetic east' if event['flow'] else 'East')+' (km)',ylabel=('Magnetic north' if event['flow'] else 'North')+' (km)')
    ax.set_aspect('equal')
    if title: ax.set_title(title,fontsize=11)


def plot_event(event, results, prefix):
    fig, axs=plt.subplots(1,3,figsize=(14,4 if event['flow'] else 7),layout='constrained')
    image_panel(axs[0],event,'Optical data + measured vectors')
    x,y=event['x'],event['physical_y'];scale=30. if event['flow'] else 2.
    step=3 if event['flow'] else 1
    arrows=axs[0].quiver(x[::step,0],x[::step,1],y[::step,0],y[::step,1],color='#18CEEF',angles='xy',scale_units='xy',scale=scale,width=.006)
    axs[0].quiverkey(arrows,.8,.95,500 if event['flow'] else 50,'500 m/s' if event['flow'] else '50 mV/m',labelpos='S',labelcolor='white')
    axs[0].scatter(x[:,0],x[:,1],s=5,color='#18CEEF')
    exported={}
    for ax,name in zip(axs[1:],['electrostatic','image_aligned']):
        image_panel(ax,event,LABELS[name]+' reconstruction')
        model,best=fit_tuned(x,event['y'],event['noise'],event['ids'],name,event['polynomial'],event['flow'],event['output_scale'])
        gx=np.linspace(-120,120,15);gy=np.linspace(-65,65,15) if event['flow'] else np.linspace(-60,400,22)
        xx,yy=np.meshgrid(gx,gy);q=np.column_stack([xx.ravel(),yy.ravel()]);pred,var=model.predict(q,True)
        if not event['flow']:
            metric=electric_output_scale(q)
            pred *= metric
            var *= metric[:,:,None]*metric[:,None,:]
        exported[name+'_mean']=pred
        exported[name+'_conditional_covariance']=var
        close=cdist(q,x).min(1)<=50
        # Distant predictions shown faintly to disclose extrapolation.
        for m,alpha in [(close,1.),(~close,.25)]:
            ax.quiver(q[m,0],q[m,1],pred[m,0],pred[m,1],color='#18CEEF',alpha=alpha,angles='xy',scale_units='xy',scale=scale,width=.004)
        ax.scatter(x[:,0],x[:,1],s=7,color='#FF714F',label='Measurements')
        ax.legend(loc='upper right',fontsize=8)
    context='Optical inversion, 2017-03-02 07:54:10 UTC; arrows: m/s' if event['flow'] else f"DASC 428 nm, {event['metadata']['utc']}; arrows: mV/m"
    fig.suptitle(context+'\nFaint arrows >50 km from measurements are extrapolations; imagery is not field truth.',fontsize=11)
    fig.savefig(ROOT/'figures'/f'{prefix}_maps.png',dpi=170)
    fig.savefig(ROOT/'figures'/f'{prefix}_maps.pdf')
    plt.close(fig)
    exported['coordinates_km']=q
    exported['distance_to_measurement_km']=cdist(q,x).min(1)
    exported['grid_shape']=np.array(xx.shape)
    np.savez_compressed(ROOT/'results'/f'{prefix}_field_map.npz',**exported)
    fig,axs=plt.subplots(1,2,figsize=(10,4 if event['flow'] else 6),layout='constrained')
    spread=np.sqrt(np.trace(exported['image_aligned_conditional_covariance'],axis1=1,axis2=2))
    disagreement=np.linalg.norm(exported['image_aligned_mean']-exported['electrostatic_mean'],axis=1)
    for ax,values,title in zip(axs,[spread,disagreement],['Conditional latent vector RMS spread','Image vs isotropic prediction difference']):
        im=ax.pcolormesh(xx,yy,values.reshape(xx.shape),shading='auto',cmap='magma')
        ax.scatter(x[:,0],x[:,1],s=7,color='cyan');ax.set_aspect('equal')
        ax.set(xlabel='East (km)',ylabel='North (km)',title=title)
        fig.colorbar(im,ax=ax,label='m/s' if event['flow'] else 'mV/m')
    fig.suptitle(event['name']+' — uncertainty diagnostics\nConditional spread omits geometry/time/systematic errors; disagreement is not measured error.',fontsize=11)
    fig.savefig(ROOT/'figures'/f'{prefix}_uncertainty.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(11,6),sharex=True,layout='constrained')
    for ax,component,key in zip(axs,['east','north'],['e','n']):
        ax.plot(event['ids'],event['physical_y'][:,0 if key=='e' else 1],'.-',color='#252525',lw=.7,label='Measured')
        for name in ['independent','electrostatic','image_aligned']:
            r=results[(results.event==event['name'])&(results.model==name)].sort_values('row')
            ax.plot(r.row,r[f'predicted_{key}'],'o-',ms=3,lw=.8,color=COLORS[name],label=LABELS[name])
        ax.set_ylabel(component+' ('+('m/s' if event['flow'] else 'mV/m')+')');ax.grid(alpha=.2)
    axs[0].legend(ncol=2,fontsize=9);axs[1].set_xlabel('Original measurement row / latitude bin')
    fig.suptitle('Nested contiguous holdout predictions — '+event['name'])
    fig.savefig(ROOT/'figures'/f'{prefix}_holdouts.png',dpi=160);plt.close(fig)


def plot_photographs():
    from PIL import Image
    names=['095125_0428','095128_0558','095130_0630']
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,stem,label in zip(axs,names,['428 nm / 09:51:25','558 nm / 09:51:28','630 nm / 09:51:30']):
        im=np.asarray(Image.open(DATA/f'PKR_20230210_{stem}.png'))
        ax.imshow(im,cmap='gray',vmin=np.quantile(im,.05),vmax=np.quantile(im,.995));ax.set_title(label+' UTC');ax.axis('off')
    fig.suptitle('Actual Poker Flat all-sky camera exposures — 10 February 2023\nSeparate monochrome filters; contrast scaled independently, not a simultaneous RGB photograph.',fontsize=11)
    fig.savefig(ROOT/'figures'/'actual_auroral_photographs.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sensitivity',action='store_true');args=parser.parse_args()
    (ROOT/'results').mkdir(exist_ok=True);(ROOT/'figures').mkdir(exist_ok=True)
    events=[radar_event(tag) for tag in EVENTS]+[rocket_event()]
    frames=[];all_selections=[]
    for e in events:
        print('Fitting',e['name'],len(e['x']),'measurements',flush=True)
        frame,selected=evaluate(e);frames.append(frame);all_selections+=selected
    predictions=pd.concat(frames,ignore_index=True)
    predictions.to_csv(ROOT/'results/predictions.csv',index=False)
    summary=predictions.groupby(['event','model']).agg(n=('row','count'),mse=('squared_error','mean'),coverage95=('covered95','mean'),direction_error_deg=('angle_degrees','mean')).reset_index()
    summary['rmse']=np.sqrt(summary.mse)
    summary.to_csv(ROOT/'results/event_metrics.csv',index=False)
    (ROOT/'results/selections.json').write_text(json.dumps(all_selections,indent=2)+'\n')
    (ROOT/'results/data_metadata.json').write_text(json.dumps([e['metadata'] for e in events],indent=2)+'\n')
    # All event images, not just the most attractive example.
    fig,axs=plt.subplots(2,3,figsize=(13,10),layout='constrained')
    for ax,e in zip(axs.flat,events[:-1]):
        image_panel(ax,e,e['metadata']['utc'][:19]);ax.scatter(e['x'][:,0],e['x'][:,1],s=16,color='#18CEEF',label='Retained radar bins');ax.legend(fontsize=7)
    fig.suptitle('Six event optical snapshots; image-only quadratic ridge in gold\nRadar vectors are five-minute processed estimates mapped to 110 km.',fontsize=13)
    fig.savefig(ROOT/'figures/six_event_geometry.png',dpi=160);plt.close(fig)
    plot_photographs()
    for e in [events[0],events[-1]]:plot_event(e,predictions,e['name'])
    radar=summary[summary.event!='isinglass'];table=radar.pivot(index='event',columns='model',values='rmse')
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for name in MODELS:
        ax.plot(range(len(table)),table[name],'o-',label=LABELS[name],color=COLORS[name],lw=1.3)
    ax.set_xticks(range(len(table)),table.index,rotation=20);ax.set_ylabel('Held-out horizontal electric-field vector RMSE (mV/m)');ax.grid(alpha=.2);ax.legend(ncol=3,fontsize=9)
    ax.set_title('Nested, guarded latitude-bin holdouts — six radar events\nThese are processed-bin predictions, not independent-track validation.')
    fig.savefig(ROOT/'figures/radar_comparison.png',dpi=170);plt.close(fig)
    if args.sensitivity:
        rows=[]
        for sigma in [50.,200.]:
            e=rocket_event(sigma=sigma);p,_=evaluate(e,['independent','electrostatic','image_aligned'])
            for name,g in p.groupby('model'):rows.append(dict(study='rocket_noise',setting=sigma,model=name,rmse=float(np.sqrt(g.squared_error.mean()))))
        for frame in [0,45,135]:
            e=rocket_event(frame=frame);p,_=evaluate(e,['image_aligned'])
            rows.append(dict(study='rocket_image_time',setting=frame,model='image_aligned',rmse=float(np.sqrt(p.squared_error.mean()))))
        # Added after resolving MATLAB datetime metadata; post-hoc temporal audit.
        for frame in [0,45,90,135]:
            e=rocket_event(frame=frame,max_time_offset=30.)
            p,_=evaluate(e,['independent','electrostatic','image_aligned'])
            for name,g in p.groupby('model'):
                rows.append(dict(study='rocket_matched_30s',setting=frame,model=name,n=len(e['x']),rmse=float(np.sqrt(g.squared_error.mean()))))
        for limit in [20.,50.]:
            for tag in EVENTS:
                e=radar_event(tag,limit)
                if len(e['x'])<9:continue
                p,_=evaluate(e,['independent','electrostatic','image_aligned'])
                for name,g in p.groupby('model'):rows.append(dict(study='radar_quality',event=tag,setting=limit,model=name,n=len(e['x']),rmse=float(np.sqrt(g.squared_error.mean()))))
        pd.DataFrame(rows).to_csv(ROOT/'results/sensitivity.csv',index=False)
    print(summary[['event','model','n','rmse','coverage95']].to_string(index=False),flush=True)

if __name__=='__main__':main()
