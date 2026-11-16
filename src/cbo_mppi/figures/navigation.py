#!/usr/bin/env python3
"""Render the Navigation figures at publication size without reevaluating controls or objectives.

The closed-loop figure uses the same frozen NPZ as the five-panel version. For
open-loop figures, vector paths are extracted from the original PDFs with
Poppler and redrawn. This only recomposes already rendered curves; it neither
recovers raw Monte Carlo values nor performs new numerical analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from . import closed_loop as original
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector_lines(pdf):
    with tempfile.TemporaryDirectory() as tmp:
        svg = Path(tmp) / "figure.svg"
        subprocess.run(["pdftocairo", "-svg", str(pdf), str(svg)], check=True)
        root = ET.parse(svg).getroot()
    paths = []
    for elem in root.iter():
        if not elem.tag.endswith('path') or elem.get('fill') != 'none':
            continue
        d = elem.get('d', '')
        if not elem.get('stroke') or not re.fullmatch(r'[ML\s\d.\-]+', d):
            continue
        parts = re.findall(r'([ML])\s+([-\d.]+)\s+([-\d.]+)', d)
        vertices = []
        for command, x, y in parts:
            if command == 'M' and vertices:
                vertices.append((np.nan, np.nan))
            vertices.append((float(x), float(y)))
        paths.append((elem, np.array(vertices)))
    return paths


def save(fig, path, vertical_bbox=None):
    # Preserve the width and element sizes; crop only vertical margins when requested.
    kwargs = {'metadata': {'CreationDate': None, 'ModDate': None}}
    if vertical_bbox is not None:
        bottom, top = vertical_bbox
        kwargs.update(bbox_inches=Bbox.from_extents(0, bottom, fig.get_figwidth(), top),
                      pad_inches=0)
    fig.savefig(path, **kwargs)
    plt.close(fig)


def open_loop(out, route_pdf, landscape_pdf):
    route_paths = [(e, v) for e, v in vector_lines(route_pdf)
                   if e.get('stroke-linecap') == 'square'
                   and (e.get('stroke-opacity') == '0.75' or e.get('stroke-width') == '1.5')
                   and np.ptp(v[:, 0]) > 100]
    landscape_paths = [(e, v) for e, v in vector_lines(landscape_pdf)
                       if e.get('stroke-width') == '1.3' and len(v) > 50]
    if len(route_paths) != 9 or len(landscape_paths) != 3:
        raise ValueError('The original figure structure has changed; review the vector extraction criteria.')
    center = next(v for e, v in route_paths if e.get('stroke-width') == '1.5')
    x0, x1 = center[0, 0], center[-1, 0]
    scale = (x1 - x0) / 8
    y0 = center[0, 1]
    fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.12))
    fig.subplots_adjust(left=.07, right=.985, bottom=.19, top=.79, wspace=.34)
    ax = axes[0]
    route_heights = [-3, -2, -1, -.5, 0, .5, 1, 2, 3]
    route_norm = colors.Normalize(vmin=-3, vmax=3)
    route_cmap = plt.get_cmap('viridis')
    for h, (_, v) in zip(route_heights, route_paths):
        ax.plot((v[:, 0]-x0)/scale-4, (v[:, 1]-y0)/scale,
                color=route_cmap(route_norm(h)), lw=1.1 if h == 0 else .8)
    for y in (-.85,.85):
        ax.add_patch(plt.Circle((0,y),.5,color='.6',alpha=.5,zorder=1))
    ax.plot(-4,0,'ks',ms=3); ax.plot(4,0,'k*',ms=6)
    ax.set(xlim=(-5,5),ylim=(-3.25,3.25),xlabel='$x$',ylabel='$y$')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([-4,0,4]); ax.set_yticks([-3,0,3])
    route_map = plt.cm.ScalarMappable(norm=route_norm, cmap=route_cmap)
    route_map.set_array([])
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0 + .12*pos.width, pos.y1 + .075,
                        .76*pos.width, .025])
    cbar = fig.colorbar(route_map, cax=cax, orientation='horizontal')
    cbar.set_ticks([-3, 0, 3])
    cbar.set_ticklabels([r'$-3$', r'$0$', r'$+3$'])
    cbar.ax.tick_params(labelsize=8, pad=1, length=2)
    cbar.ax.text(-.07, .5, r'$h$', transform=cbar.ax.transAxes,
                 ha='right', va='center', fontsize=9)
    ax = axes[1]
    # The original figure normalizes each curve independently to [0, 1].
    all_v = np.concatenate([v for e,v in landscape_paths])
    xmin,xmax = np.nanmin(all_v[:,0]),np.nanmax(all_v[:,0])
    ymin,ymax = np.nanmin(all_v[:,1]),np.nanmax(all_v[:,1])
    for (elem,v),label,color,ls in zip(landscape_paths,
            [r'$\bar J$',r'$\widehat F^\Pi_{15}$',r'$\widehat F^\Pi_{300}$'],
            ['#d62728','#009E73','#0072B2'],['-', '--', '-']):
        x=(v[:,0]-xmin)/(xmax-xmin)*6-3
        y=(v[:,1]-ymin)/(ymax-ymin)
        ax.plot(x,y,color=color,ls=ls,lw=1.1,label=label)
        index=np.nanargmin(y);ax.plot(x[index],y[index],'o',color=color,ms=3)
    ax.set(xlim=(-3.3,3.3),ylim=(-.04,1.04),xlabel=r'route height $h$',ylabel='normalized objective')
    ax.set_xticks([-3,0,3]);ax.set_yticks([0,.5,1])
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.04),ncol=3,
              handlelength=1.3,columnspacing=.65,handletextpad=.3,borderpad=0,frameon=False)
    for ax in axes:
        ax.grid(alpha=.18);ax.tick_params(pad=2)
        ax.xaxis.labelpad=2;ax.yaxis.labelpad=2
    # Match the plot-frame heights while strictly preserving the left panel's aspect ratio.
    pos=axes[0].get_position();right=axes[1].get_position()
    axes[1].set_position([right.x0,pos.y0,right.width,pos.height])
    output=out/'navigation_open_loop.pdf';save(fig,output,vertical_bbox=(4/72,142/72))
    return {'inputs':{(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)):sha(p)
                      for p in (route_pdf,landscape_pdf)},
            'output_sha256':sha(output),'route_paths':9,'landscape_paths':3,
            'source_kind':'existing PDF vector paths; no new Monte Carlo evaluation',
            'canvas_inches':[6.1,2.12],'published_bbox_inches':[0,4/72,6.1,142/72],
            'minimum_font_pt':8.5}


def open_loop_arrays(out, route_data, landscape_data):
    with np.load(route_data, allow_pickle=False) as route:
        h_values = np.asarray(route['h_values'])
        trajectories = np.asarray(route['trajectories'])
        centers = np.asarray(route['obstacle_centers'])
        radii = np.asarray(route['obstacle_radii'])
        start = np.asarray(route['start'])
        goal = np.asarray(route['goal'])
    with np.load(landscape_data, allow_pickle=False) as landscape:
        heights = np.asarray(landscape['heights'])
        nominal = np.asarray(landscape['nominal_costs'])
        taus = np.asarray(landscape['taus'])
        free_energies = np.asarray(landscape['free_energies'])

    fig, axes = plt.subplots(1, 2, figsize=(6.1, 2.12))
    fig.subplots_adjust(left=.07, right=.985, bottom=.19, top=.79, wspace=.34)
    ax = axes[0]
    route_norm = colors.Normalize(vmin=float(h_values.min()), vmax=float(h_values.max()))
    route_cmap = plt.get_cmap('viridis')
    for h, trajectory in zip(h_values, trajectories):
        ax.plot(trajectory[:, 0], trajectory[:, 1],
                color=route_cmap(route_norm(h)), lw=1.1 if h == 0 else .8)
    for center, radius in zip(centers, radii):
        ax.add_patch(plt.Circle(center, radius, color='.6', alpha=.5, zorder=1))
    ax.plot(*start, 'ks', ms=3); ax.plot(*goal, 'k*', ms=6)
    ax.set(xlim=(-5,5), ylim=(-3.25,3.25), xlabel='$x$', ylabel='$y$')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([-4,0,4]); ax.set_yticks([-3,0,3])
    route_map = plt.cm.ScalarMappable(norm=route_norm, cmap=route_cmap)
    route_map.set_array([])
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0 + .12*pos.width, pos.y1 + .075,
                        .76*pos.width, .025])
    cbar = fig.colorbar(route_map, cax=cax, orientation='horizontal')
    cbar.set_ticks([-3, 0, 3]); cbar.set_ticklabels([r'$-3$', r'$0$', r'$+3$'])
    cbar.ax.tick_params(labelsize=8, pad=1, length=2)
    cbar.ax.text(-.07, .5, r'$h$', transform=cbar.ax.transAxes,
                 ha='right', va='center', fontsize=9)

    ax = axes[1]
    def normalized(values):
        values = np.where(np.isfinite(values), values, np.nan)
        lo, hi = np.nanmin(values), np.nanmax(values)
        return (values-lo)/(hi-lo) if hi > lo else values-lo
    curves = [(nominal, r'$\bar J$', '#d62728', '-')]
    palette = {15: ('#009E73', '--'), 300: ('#0072B2', '-')}
    for tau, values in zip(taus, free_energies):
        color, style = palette.get(int(tau), ('#0072B2', '-'))
        curves.append((values, rf'$\widehat F^\Pi_{{{tau:g}}}$', color, style))
    for values, label, color, style in curves:
        y = normalized(values)
        ax.plot(heights, y, color=color, ls=style, lw=1.1, label=label)
        index = int(np.nanargmin(y)); ax.plot(heights[index], y[index], 'o', color=color, ms=3)
    ax.set(xlim=(-3.3,3.3), ylim=(-.04,1.04), xlabel=r'route height $h$', ylabel='normalized objective')
    ax.set_xticks([-3,0,3]); ax.set_yticks([0,.5,1])
    ax.legend(loc='lower center', bbox_to_anchor=(.5,1.04), ncol=3,
              handlelength=1.3, columnspacing=.65, handletextpad=.3,
              borderpad=0, frameon=False)
    for ax in axes:
        ax.grid(alpha=.18); ax.tick_params(pad=2)
        ax.xaxis.labelpad=2; ax.yaxis.labelpad=2
    pos=axes[0].get_position(); right=axes[1].get_position()
    axes[1].set_position([right.x0,pos.y0,right.width,pos.height])
    output=out/'navigation_open_loop.pdf'; save(fig,output,vertical_bbox=(4/72,142/72))
    return {'inputs': {str(route_data): sha(route_data), str(landscape_data): sha(landscape_data)},
            'output_sha256': sha(output), 'source_kind': 'fresh numerical arrays'}


def closed_loop(data_file,out):
    keys = (
        'navigation_mppi_low', 'navigation_mppi_high',
        'navigation_cbo_mppi_low', 'navigation_cbo_mppi_high',
        'navigation_cbo_mpc',
    )
    with original.load_data(data_file) as data:
        batches = [np.array(original.states(data, key, 80)) for key in keys]
    assert len(batches)==5 and all(len(batch)==100 for batch in batches)
    fig,axes=plt.subplots(1,3,figsize=(6.1,1.72))
    fig.subplots_adjust(left=.065,right=.99,bottom=.31,top=.85,wspace=.26)
    for ax,title in zip(axes,['MPPI','CBO-MPC','CBO-MPPI']):
        original.nav_panel(ax,[(0,.85),(0,-.85)],[.5,.5],(-4,0),(4,0),
                           [(-5.2,5.2),(-2.65,2.65)])
        ax.set_title(title,pad=4,fontsize=10)
        ax.set_xticks([-4,0,4]);ax.set_yticks([-2,0,2])
        ax.xaxis.labelpad=1;ax.yaxis.labelpad=1;ax.tick_params(pad=2)
    for ax,low,high in [(axes[0],batches[0],batches[1]),(axes[2],batches[2],batches[3])]:
        # Draw low-temperature dashed curves on top so high-temperature solid curves remain visible in the gaps.
        for x in high:ax.plot(x[:,0],x[:,1],color='#0072B2',lw=.8,alpha=.45,zorder=2)
        for x in low:ax.plot(x[:,0],x[:,1],color='#D55E00',ls=(0,(4,3)),lw=.8,alpha=.55,zorder=3)
    for x in batches[4]:axes[1].plot(x[:,0],x[:,1],color='#555555',lw=.8,alpha=.5,zorder=2)
    handles=[Line2D([],[],color='#D55E00',ls=(0,(4,3)),label=r'$\tau=15$'),
             Line2D([],[],color='#0072B2',label=r'$\tau=300$'),
             Line2D([],[],color='#555555',label=r'CBO-MPC ($\tau$-independent)')]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,
               bbox_to_anchor=(.5,-.015),handlelength=2,columnspacing=1)
    output=out/'navigation_three_panel.pdf';save(fig,output,vertical_bbox=(6/72,112/72))
    return {'input':str(data_file.relative_to(ROOT)) if data_file.is_relative_to(ROOT) else str(data_file),
            'input_sha256':sha(data_file),'keys':list(keys),
            'output_sha256':sha(output),'runs':500,
            'states_per_run':81,'canvas_inches':[6.1,1.72],
            'published_bbox_inches':[0,6/72,6.1,112/72],
            'ordering':'MPPI low/high, CBO-MPPI low/high, CBO-MPC',
            'selection':'all performance seeds 51000..51099 in every condition, no outcome filtering'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-file',type=Path,default=original.DEFAULT_DATA)
    p.add_argument('--route-source',type=Path,
                   default=ROOT/'data/frozen/figures/navigation/route_height_trajectories.pdf')
    p.add_argument('--landscape-source',type=Path,
                   default=ROOT/'data/frozen/figures/navigation/case1_landscape.pdf')
    p.add_argument('--route-data',type=Path)
    p.add_argument('--landscape-data',type=Path)
    p.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/figures')
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,
                         'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,
                         'font.family':'serif',
                         'text.latex.preamble':r'\usepackage[T1]{fontenc}\usepackage{times}',
                         'pdf.fonttype':42,'text.usetex':True,
                         'axes.unicode_minus':False})
    report={'generators':{str(p.relative_to(ROOT)):sha(p) for p in
                          (Path(__file__).resolve(), Path(original.__file__).resolve())},
            'open_loop':(open_loop_arrays(a.output_dir,a.route_data,a.landscape_data)
                         if a.route_data is not None and a.landscape_data is not None
                         else open_loop(a.output_dir,a.route_source,a.landscape_source)),
            'closed_loop':closed_loop(a.data_file,a.output_dir)}
    report_path = a.output_dir/'navigation_provenance.json'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
