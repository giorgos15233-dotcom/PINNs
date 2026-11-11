"""Build field profiles, near-field maps, and thesis comparison plots with pvpython."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import numpy as np
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "navier-stokes": "Navier–Stokes  ·  200,000 steps",
    "k-epsilon": "Standard k–ε  ·  200,000 steps",
    "k-epsilon_wf_1": "Wall-function k–ε  ·  44,000 steps",
    "Ver2_A": "Ver2 A  ·  400,000 steps",
    "Ver2_B": "Ver2 B  ·  115,000 steps",
    "Ver2_D_yplus": "Ver2 D  ·  step unknown",
}
LABELS = {"p": "Static pressure p", "u": "Streamwise velocity u",
          "v": "Transverse velocity v", "k": "Turbulent kinetic energy k",
          "ep": "Dissipation rate ε"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlelocation": "left", "figure.facecolor": "white"})

def load_field(path):
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    mesh = reader.GetOutput()
    xy = vtk_to_numpy(mesh.GetPoints().GetData()).copy()
    arrays = {key: vtk_to_numpy(mesh.GetPointData().GetArray(key)).ravel().copy()
              for key in LABELS if mesh.GetPointData().HasArray(key)}
    assert all(np.isfinite(a).all() and len(a) == len(xy) for a in arrays.values())
    return xy, arrays

def plot_field(folder, region, source, title):
    xy, arrays = load_field(source)
    fig, axs = plt.subplots((len(arrays)+1)//2, 2, figsize=(10.5, 2.65*((len(arrays)+1)//2)+0.6),
                            squeeze=False, constrained_layout=True)
    fig.suptitle(f"{title}  |  {region.replace('_',' ').title()}", fontsize=15)
    for ax, (key, a) in zip(axs.flat, arrays.items()):
        ax.set_title(LABELS[key])
        if region == "boundary":
            for upper, color in [(True,"#036b87"),(False,"#d56825")]:
                mask = (xy[:,1] >= 0) == upper
                order = np.flatnonzero(mask)[np.argsort(xy[mask,0])]
                ax.plot(xy[order,0], a[order], color=color, lw=1.2, label="Upper" if upper else "Lower")
            if ax is axs.flat[0]: ax.legend(frameon=False, ncol=2, fontsize=8)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2,2), useOffset=False)
            ax.grid(alpha=.16)
        else:
            low, high = float(a.min()), float(a.max())
            if high == low:
                ax.scatter(xy[:,0],xy[:,1],s=2,color="#22798a",linewidths=0)
                ax.text(.98,.95,f"Uniform: {low:.3g}",transform=ax.transAxes,ha="right",va="top")
            else:
                signed = low < 0 < high
                if signed: low, high = -max(abs(low),abs(high)), max(abs(low),abs(high))
                artist = ax.scatter(xy[:,0], xy[:,1], c=a, vmin=low, vmax=high,
                                    cmap="RdBu_r" if signed else "viridis", s=2.4, linewidths=0)
                bar=fig.colorbar(artist,ax=ax,shrink=.77,pad=.025)
                bar.locator=MaxNLocator(4); bar.formatter=FormatStrFormatter("%.2g"); bar.update_ticks()
            ax.set_aspect("equal");ax.set_ylabel("y/c")
        ax.set_xlabel("x/c")
    for ax in list(axs.flat)[len(arrays):]: ax.remove()
    fig.savefig(folder/(region+".png"),dpi=125);plt.close(fig)

def comparison():
    with (ROOT/"results/v1.csv").open(encoding="utf-8-sig") as f: data=list(csv.DictReader(f))
    labels=[("k-epsilon","velocity_gradient","Standard k–ε"),("k-epsilon_wf_1","velocity_gradient","Wall function · gradient"),
            ("k-epsilon_wf_1","surface_u_tau_squared","Wall function · uτ²"),("navier-stokes","velocity_gradient","Navier–Stokes")]
    fig,(drag,lift)=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for i,(model,method,label) in enumerate(labels):
        row=next(r for r in data if r["model"]==model and r["method"]==method)
        p,v=float(row["Cd_pressure"]),float(row["Cd_friction"])
        drag.barh(i,p,color="#396882",label="Pressure" if not i else None)
        drag.barh(i,v,left=p,color="#e4a65c",label="Friction" if not i else None)
        lift.scatter(abs(float(row["Cl"])),i,color="#396882")
    drag.axvline(.0065,ls="--",color="#342e2a",label="Eppler reference")
    drag.set_xlim(0,.0071);drag.set_yticks(range(4),[x[2] for x in labels])
    drag.set_xlabel("Drag coefficient");lift.set_xlabel("Absolute lift coefficient")
    lift.set_xscale("log");lift.set_yticks(range(4),[])
    for ax in (drag,lift):ax.set_ylim(3.6,-.6);ax.grid(axis="x",alpha=.16)
    drag.legend(loc="upper center",bbox_to_anchor=(.5,-.2),ncol=3,fontsize=8,frameon=False)
    fig.suptitle("NACA 0012 at zero incidence  |  Re = 10⁶")
    fig.savefig(ROOT/"figures/comparison.png",dpi=155);plt.close(fig)
    with (ROOT/"results/loss_history.csv").open(encoding="utf8") as f: hist=list(csv.DictReader(f))
    fig,axs=plt.subplots(1,3,figsize=(10,3.5),constrained_layout=True)
    for ax,m in zip(axs,["k-epsilon","k-epsilon_wf_1","navier-stokes"]):
        series=[(int(r["step"])/1000,float(r["loss_aggregated"])) for r in hist if r["model"]==m]
        ax.semilogy(*zip(*series),color="#396882",lw=1);ax.set_title(m);ax.set_xlabel("Training steps (thousands)")
        ax.grid(alpha=.18)
    axs[0].set_ylabel("Aggregated training loss");fig.suptitle("Training histories  |  Re = 10⁶")
    fig.savefig(ROOT/"figures/loss.png",dpi=155);plt.close(fig)

if __name__ == "__main__":
    for name,title in CASES.items():
        folder=ROOT/"fields"/name
        for region in ["boundary","near_field"]:
            source=folder/(region+"_1.vtp")
            if source.is_file(): plot_field(folder,region,source,title)
    comparison()
