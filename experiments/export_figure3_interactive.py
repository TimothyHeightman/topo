#!/usr/bin/env python3
"""Export the banked trained-FNQS pair event as an interactive HTML fragment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "experiments" / "results" / "figure3_fnqs_pair"
DEFAULT_OUTPUT = PROJECT_ROOT / "visualizations" / "neutral-pair-creation.html"


def compact(values: np.ndarray, digits: int = 4) -> list:
    return np.round(values, digits).tolist()


def load_worldlines(path: Path) -> dict[str, list[list[float]]]:
    branches: dict[str, list[list[float]]] = {"neg": [], "pos": []}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            charge = float(row["charge"])
            key = "pos" if charge > 0.0 else "neg"
            branches[key].append(
                [
                    round(float(row["tau"]), 7),
                    round(float(row["u"]), 6),
                    round(float(row["v"]), 6),
                ]
            )
    return branches


def fragment(data: dict) -> str:
    payload = json.dumps(data, separators=(",", ":"))
    return f'''<div id="trained-fnqs-pair">
  <style>
    #trained-fnqs-pair {{ color:var(--foreground); font-family:ui-sans-serif,system-ui,sans-serif; width:100%; padding:2px 0 4px; }}
    #trained-fnqs-pair * {{ box-sizing:border-box; }}
    #trained-fnqs-pair .tfp-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:16px; margin:0 2px 7px; flex-wrap:wrap; }}
    #trained-fnqs-pair .tfp-title {{ font-family:ui-serif,Georgia,serif; font-size:18px; font-weight:600; }}
    #trained-fnqs-pair .tfp-legend {{ display:flex; gap:14px; flex-wrap:wrap; }}
    #trained-fnqs-pair button {{ border:0; background:transparent; color:var(--foreground); font:inherit; font-size:12px; padding:2px 0; display:inline-flex; align-items:center; gap:6px; cursor:pointer; }}
    #trained-fnqs-pair button[aria-pressed="false"] {{ opacity:.42; }}
    #trained-fnqs-pair .tfp-swatch {{ width:18px; height:3px; display:inline-block; }}
    #trained-fnqs-pair .tfp-swatch.neg {{ background:var(--viz-series-5); }}
    #trained-fnqs-pair .tfp-swatch.pos {{ background:var(--viz-series-3); }}
    #trained-fnqs-pair .tfp-panels {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
    #trained-fnqs-pair .tfp-panel {{ min-width:0; position:relative; }}
    #trained-fnqs-pair svg {{ width:100%; display:block; overflow:visible; }}
    #trained-fnqs-pair text {{ fill:var(--foreground); font-family:ui-serif,Georgia,serif; font-size:12px; }}
    #trained-fnqs-pair .tfp-axis path,#trained-fnqs-pair .tfp-axis line {{ stroke:var(--border); stroke-width:.75; }}
    #trained-fnqs-pair .axis-title {{ font-size:13px; font-style:italic; }}
    #trained-fnqs-pair .panel-label {{ font-size:15px; font-weight:600; }}
    #trained-fnqs-pair rect[data-chart-frame] {{ fill:none; stroke:var(--border); stroke-width:.8; pointer-events:none; }}
    #trained-fnqs-pair .tfp-control {{ display:grid; grid-template-columns:auto minmax(130px,340px) 58px; justify-content:center; align-items:center; gap:10px; margin-top:5px; font-family:ui-serif,Georgia,serif; font-size:13px; }}
    #trained-fnqs-pair input {{ width:100%; accent-color:var(--viz-series-3); }}
    #trained-fnqs-pair output {{ font-variant-numeric:tabular-nums; }}
    #trained-fnqs-pair .tooltip {{ position:absolute; pointer-events:none; opacity:0; z-index:4; padding:5px 7px; border:1px solid var(--border); background:var(--popover); color:var(--popover-foreground); font-size:12px; }}
    @media(max-width:560px) {{ #trained-fnqs-pair .tfp-panels {{ grid-template-columns:1fr; gap:8px; }} }}
  </style>
  <div class="tfp-head">
    <div class="tfp-title">Neutral-pair creation in a trained FNQS</div>
    <div class="tfp-legend" aria-label="Nodal-charge legend">
      <button type="button" data-charge="neg" aria-pressed="true"><span class="tfp-swatch neg"></span>charge −1</button>
      <button type="button" data-charge="pos" aria-pressed="true"><span class="tfp-swatch pos"></span>charge +1</button>
    </div>
  </div>
  <div class="tfp-panels">
    <div class="tfp-panel"><svg id="tfp-u" role="img" aria-label="Trained FNQS fidelity projected onto local coordinate u and time"></svg></div>
    <div class="tfp-panel"><svg id="tfp-v" role="img" aria-label="Trained FNQS fidelity projected onto local coordinate v and time"></svg></div>
  </div>
  <div class="tfp-control">
    <label for="tfp-time">selected time&nbsp; t/T</label>
    <input id="tfp-time" type="range" min="0" max="{len(data['tau']) - 1}" step="1" value="{len(data['tau']) // 2}" aria-label="Selected normalized time">
    <output id="tfp-output" for="tfp-time"></output>
  </div>
  <div class="tooltip" role="tooltip"></div>
  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <script>
    (() => {{
      const root=document.getElementById("trained-fnqs-pair");
      const data={payload};
      const slider=root.querySelector("#tfp-time"), output=root.querySelector("#tfp-output"), tooltip=root.querySelector(".tooltip");
      const state={{index:Number(slider.value),neg:true,pos:true}};
      const colors={{neg:"var(--viz-series-5)",pos:"var(--viz-series-3)"}};
      function raster(values,nx,ny) {{
        const canvas=document.createElement("canvas"); canvas.width=nx; canvas.height=ny;
        const context=canvas.getContext("2d"), image=context.createImageData(nx,ny);
        for(let j=0;j<ny;j+=1) for(let i=0;i<nx;i+=1) {{
          const value=values[(ny-1-j)*nx+i], c=d3.rgb(d3.interpolateMagma((value+3)/3)), o=4*(j*nx+i);
          image.data[o]=c.r; image.data[o+1]=c.g; image.data[o+2]=c.b; image.data[o+3]=255;
        }}
        context.putImageData(image,0,0); return canvas.toDataURL("image/png");
      }}
      function drawPanel(svgId,key,label,panelLabel) {{
        const svg=d3.select(root.querySelector(svgId)), panel=svg.node().parentElement;
        const width=Math.max(320,Math.floor(panel.getBoundingClientRect().width)), height=width<500?Math.min(290,Math.round(width*.88)):Math.min(390,Math.round(width*.76));
        const margin={{top:24,right:20,bottom:50,left:62}}, iw=width-margin.left-margin.right, ih=height-margin.top-margin.bottom;
        const coordinate=data[key], x=d3.scaleLinear().domain(d3.extent(coordinate)).range([margin.left,margin.left+iw]);
        const y=d3.scaleLinear().domain(d3.extent(data.tau)).range([margin.top+ih,margin.top]);
        svg.attr("viewBox",`0 0 ${{width}} ${{height}}`).attr("height",height); svg.selectAll("*").remove();
        const clipId=`tfp-${{key}}-clip`;
        svg.append("defs").append("clipPath").attr("id",clipId).append("rect").attr("x",margin.left).attr("y",margin.top).attr("width",iw).attr("height",ih);
        svg.append("text").attr("class","panel-label").attr("x",2).attr("y",15).text(panelLabel);
        svg.append("image").attr("x",margin.left).attr("y",margin.top).attr("width",iw).attr("height",ih).attr("preserveAspectRatio","none")
          .attr("href",raster(data[`logF_${{key}}`],coordinate.length,data.tau.length));
        const layer=svg.append("g").attr("clip-path",`url(#${{clipId}})`);
        ["neg","pos"].forEach(branch=>{{
          const values=data.branches[branch].map(d=>({{tau:d[0],coordinate:key==="u"?d[1]:d[2]}}));
          const line=d3.line().x(d=>x(d.coordinate)).y(d=>y(d.tau));
          const group=layer.append("g").attr("data-branch",branch).style("display",state[branch]?null:"none");
          group.append("path").datum(values).attr("d",line).attr("fill","none").attr("stroke","var(--background)").attr("stroke-width",5);
          group.append("path").datum(values).attr("d",line).attr("fill","none").attr("stroke",colors[branch]).attr("stroke-width",2.6);
        }});
        layer.append("line").attr("data-selected","true").attr("x1",margin.left).attr("x2",margin.left+iw).attr("y1",y(data.tau[state.index])).attr("y2",y(data.tau[state.index])).attr("stroke","var(--foreground)").attr("stroke-width",1.1).attr("stroke-dasharray","4 3");
        const xa=svg.append("g").attr("class","tfp-axis").attr("transform",`translate(0,${{margin.top+ih}})`).call(d3.axisBottom(x).ticks(width<420?4:5).tickSizeOuter(0));
        const ya=svg.append("g").attr("class","tfp-axis").attr("transform",`translate(${{margin.left}},0)`).call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".3f")).tickSizeOuter(0));
        svg.append("rect").attr("data-chart-frame","true").attr("x",margin.left).attr("y",margin.top).attr("width",iw).attr("height",ih);
        svg.append("text").attr("class","axis-title").attr("data-axis","x").attr("x",margin.left+iw/2).attr("y",height-9).attr("text-anchor","middle").text(`${{label}} (rad)`);
        svg.append("text").attr("class","axis-title").attr("data-axis","y").attr("transform",`translate(17,${{margin.top+ih/2}}) rotate(-90)`).attr("text-anchor","middle").text("t/T");
        svg.append("rect").attr("data-chart-hit","true").attr("x",margin.left).attr("y",margin.top).attr("width",iw).attr("height",ih).attr("fill","transparent")
          .on("pointermove",event=>{{ const [mx,my]=d3.pointer(event,svg.node()), ci=Math.max(0,Math.min(coordinate.length-1,Math.round((mx-margin.left)/iw*(coordinate.length-1)))), ti=Math.max(0,Math.min(data.tau.length-1,Math.round((margin.top+ih-my)/ih*(data.tau.length-1)))), [px,py]=d3.pointer(event,root); tooltip.style.left=`${{px+9}}px`; tooltip.style.top=`${{py-28}}px`; tooltip.style.opacity=1; tooltip.textContent=`${{label}} = ${{coordinate[ci].toFixed(3)}} rad,  t/T = ${{data.tau[ti].toFixed(4)}},  log₁₀F = ${{data[`logF_${{key}}`][ti*coordinate.length+ci].toFixed(2)}}`; }}).on("pointerleave",()=>{{tooltip.style.opacity=0;}});
      }}
      function draw() {{ output.textContent=data.tau[state.index].toFixed(4); drawPanel("#tfp-u","u","u","(a)"); drawPanel("#tfp-v","v","v","(b)"); }}
      slider.addEventListener("input",()=>{{state.index=Number(slider.value); draw();}});
      root.querySelectorAll("button[data-charge]").forEach(button=>button.addEventListener("click",()=>{{const key=button.dataset.charge; state[key]=!state[key]; button.setAttribute("aria-pressed",String(state[key])); draw();}}));
      new ResizeObserver(()=>draw()).observe(root.querySelector(".tfp-panels")); draw();
    }})();
  </script>
</div>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    with np.load(RESULTS / "local_profiles.npz", allow_pickle=False) as bank:
        time_slice = slice(None, None, 4)
        coordinate_slice = slice(None, None, 4)
        log_infidelity_u = bank["log_infidelity_u"][time_slice, coordinate_slice]
        log_infidelity_v = bank["log_infidelity_v"][time_slice, coordinate_slice]
        data = {
            "tau": compact(bank["tau"][time_slice], 7),
            "u": compact(bank["u"][coordinate_slice], 5),
            "v": compact(bank["v"][coordinate_slice], 5),
            "logF_u": compact(
                np.clip(
                    np.log10(np.maximum(1.0 - 10.0**log_infidelity_u, 1.0e-3)),
                    -3.0,
                    0.0,
                ).reshape(-1),
                3,
            ),
            "logF_v": compact(
                np.clip(
                    np.log10(np.maximum(1.0 - 10.0**log_infidelity_v, 1.0e-3)),
                    -3.0,
                    0.0,
                ).reshape(-1),
                3,
            ),
            "branches": load_worldlines(RESULTS / "event_worldlines.csv"),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(fragment(data) + "\n", encoding="utf-8")
    print(args.output)
    print(f"{args.output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
