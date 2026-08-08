"""Assemble a self-contained HTML report with embedded figures."""
import os, base64
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")


def data_uri(name):
    with open(os.path.join(FIG, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


IMG = {k: data_uri(v) for k, v in {
    "traj": "verify_traj_2de4901f.png",
    "srate": "srate_per_trajectory.png",
    "tau": "tau_sweep.png",
    "trade": "tradeoff_srate_muitas.png",
}.items()}

HTML = f"""<title>MAT-SUM on Akkodis Paris trajectories</title>
<style>
:root {{
  --paper:#f6f8fb; --ink:#18202c; --muted:#5a6675; --line:#dbe2ec;
  --card:#ffffff; --accent:#1f7ab0; --accent-soft:#e3eef6; --hot:#d1621b;
  --good:#2f7d4f; --shadow:0 1px 2px rgba(20,30,45,.06),0 8px 24px rgba(20,30,45,.05);
}}
@media (prefers-color-scheme:dark) {{
  :root {{ --paper:#0e1620; --ink:#e8edf4; --muted:#9aa7b6; --line:#243244;
    --card:#141e2b; --accent:#4aa3d8; --accent-soft:#16283a; --hot:#e08a4e;
    --good:#5cbf87; --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35); }}
}}
:root[data-theme="dark"] {{ --paper:#0e1620; --ink:#e8edf4; --muted:#9aa7b6; --line:#243244;
  --card:#141e2b; --accent:#4aa3d8; --accent-soft:#16283a; --hot:#e08a4e; --good:#5cbf87;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35); }}
:root[data-theme="light"] {{ --paper:#f6f8fb; --ink:#18202c; --muted:#5a6675; --line:#dbe2ec;
  --card:#ffffff; --accent:#1f7ab0; --accent-soft:#e3eef6; --hot:#d1621b; --good:#2f7d4f;
  --shadow:0 1px 2px rgba(20,30,45,.06),0 8px 24px rgba(20,30,45,.05); }}

* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  line-height:1.6; -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:760px; margin:0 auto; padding:0 24px 96px; }}
.mono {{ font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace; }}

header {{ padding:64px 0 40px; border-bottom:1px solid var(--line); margin-bottom:40px; }}
.eyebrow {{ font-family:ui-monospace,Menlo,monospace; font-size:12px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--accent); margin:0 0 18px; }}
h1 {{ font-size:clamp(30px,5vw,44px); line-height:1.08; letter-spacing:-.02em;
  font-weight:800; margin:0 0 16px; text-wrap:balance; }}
.lede {{ font-size:19px; color:var(--muted); margin:0; max-width:60ch; }}
.meta {{ display:flex; flex-wrap:wrap; gap:8px 20px; margin-top:26px;
  font-size:13px; color:var(--muted); font-family:ui-monospace,Menlo,monospace; }}
.meta b {{ color:var(--ink); font-weight:600; }}

h2 {{ font-size:13px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted);
  font-family:ui-monospace,Menlo,monospace; margin:56px 0 8px; }}
h2 .n {{ color:var(--accent); }}
h3 {{ font-size:21px; letter-spacing:-.01em; margin:6px 0 12px; }}
p {{ margin:0 0 16px; }}
a {{ color:var(--accent); }}
strong {{ font-weight:650; }}

.kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:8px 0 8px; }}
.kpi {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:18px 16px; box-shadow:var(--shadow); }}
.kpi .v {{ font-size:30px; font-weight:800; letter-spacing:-.02em;
  font-variant-numeric:tabular-nums; }}
.kpi .l {{ font-size:12.5px; color:var(--muted); margin-top:4px; }}
@media (max-width:640px) {{ .kpis {{ grid-template-columns:repeat(2,1fr); }} }}

.callout {{ background:var(--accent-soft); border:1px solid var(--line);
  border-left:3px solid var(--accent); border-radius:12px; padding:20px 22px; margin:22px 0; }}
.callout h3 {{ margin-top:0; }}

figure {{ margin:26px 0; }}
figure img {{ width:100%; height:auto; display:block; border:1px solid var(--line);
  border-radius:12px; background:var(--card); box-shadow:var(--shadow); }}
figcaption {{ font-size:13.5px; color:var(--muted); margin-top:10px; }}
figcaption b {{ color:var(--ink); font-weight:600; }}

.tablewrap {{ overflow-x:auto; margin:18px 0; border:1px solid var(--line);
  border-radius:12px; }}
table {{ width:100%; border-collapse:collapse; font-size:14.5px; }}
th,td {{ padding:11px 14px; text-align:right; border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums; white-space:nowrap; }}
th:first-child,td:first-child {{ text-align:left; }}
thead th {{ font-family:ui-monospace,Menlo,monospace; font-size:12px; letter-spacing:.04em;
  text-transform:uppercase; color:var(--muted); background:var(--card); }}
tbody tr:last-child td {{ border-bottom:none; }}
tr.hi td {{ background:color-mix(in srgb, var(--good) 12%, transparent); font-weight:600; }}
.pill {{ display:inline-block; padding:2px 9px; border-radius:999px; font-size:12px;
  font-weight:600; font-family:ui-monospace,Menlo,monospace; }}
.pill.good {{ color:var(--good); background:color-mix(in srgb,var(--good) 15%,transparent); }}
.pill.warn {{ color:var(--hot); background:color-mix(in srgb,var(--hot) 15%,transparent); }}

ul {{ padding-left:20px; }} li {{ margin:6px 0; }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
@media (max-width:640px) {{ .cols {{ grid-template-columns:1fr; }} }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:18px 20px; box-shadow:var(--shadow); }}
.card h3 {{ margin-top:0; font-size:16px; }}
code {{ font-family:ui-monospace,Menlo,monospace; font-size:.88em;
  background:var(--accent-soft); padding:1px 6px; border-radius:5px; }}
pre {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:16px 18px; overflow-x:auto; font-size:13px; line-height:1.5; }}
footer {{ margin-top:64px; padding-top:24px; border-top:1px solid var(--line);
  color:var(--muted); font-size:13px; }}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">Method application · mobility data</p>
  <h1>Summarizing the Akkodis GPS trajectories with MAT-SUM</h1>
  <p class="lede">Applying the location-centric semantic summarization method
  (Pugliese, Lettich, Pinelli &amp; Renso, SIGSPATIAL&nbsp;'23) to 20 GPS trajectories
  in Paris — enriching the city with OpenStreetMap semantics, then compressing each
  trajectory into a handful of representative semantic locations.</p>
  <div class="meta">
    <span><b>Dataset</b> Akkodis · Paris · 4039 pts</span>
    <span><b>Config</b> squares res17 · all aspects · τ=0.9</span>
    <span><b>Metrics</b> S_rate + MUITAS</span>
  </div>
</header>

<section>
  <div class="kpis">
    <div class="kpi"><div class="v">20</div><div class="l">trajectories<br>4039 GPS points</div></div>
    <div class="kpi"><div class="v">4201</div><div class="l">semantic locations<br>from OSM enrichment</div></div>
    <div class="kpi"><div class="v">0.74</div><div class="l">S_rate at τ=0.9<br>(compression)</div></div>
    <div class="kpi"><div class="v">0.99</div><div class="l">MUITAS at τ=0.9<br>(semantics kept)</div></div>
  </div>
</section>

<section>
  <h2><span class="n">01</span> &nbsp;What was done</h2>
  <p>MAT-SUM was applied end-to-end to the Akkodis <code>points_data.csv</code>. The area
  the trajectories move through is tessellated into square cells, each cell is enriched with
  the POIs, land-use and public-transport features it contains (from OpenStreetMap), cells
  that share the same semantic context are dissolved into <strong>semantic locations</strong>,
  and every trajectory is then rewritten as a temporally-weighted sequence of the semantic
  locations it visits and summarized by merging locations whose semantic contexts are similar
  (cosine similarity ≥ τ).</p>
  <p>The implementation is a clean, self-contained port of the reference code
  (<code>github.com/chiarap2/MAT-Sum</code>), adapted to the Akkodis schema. The
  <strong>algorithm is unchanged</strong>; only the surrounding plumbing was modernised so it
  runs offline and reproducibly on this data.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>Step</th><th>Reference repo</th><th>Here</th></tr></thead>
    <tbody>
      <tr><td>Tessellation</td><td>tesspy</td><td>mercantile quadkey grid (identical)</td></tr>
      <tr><td>OSM enrichment</td><td>osmnx / live Overpass</td><td>offline Île-de-France PBF via pyrosm</td></tr>
      <tr><td>Category labels</td><td>CSV + FastText fallback</td><td>CSV map (unmapped dropped, ~4%)</td></tr>
      <tr><td>Config</td><td>168 JSON files</td><td>one CONFIG dict</td></tr>
    </tbody>
  </table></div>
  <p style="font-size:13.5px;color:var(--muted)">The live Overpass servers were overloaded at
  the time (HTTP&nbsp;504 / timeouts even on trivial queries), so enrichment was switched to a
  one-time Geofabrik PBF extract parsed locally — same tag dictionary, same label map,
  deterministic output.</p>
</section>

<section>
  <h2><span class="n">02</span> &nbsp;The area, enriched</h2>
  <p>The 20 trajectories span a bounding box of roughly 9.4&nbsp;×&nbsp;16.9&nbsp;km over
  central Paris, tessellated into <strong>4437</strong> square cells at zoom&nbsp;17 (≈150&nbsp;m).
  From OpenStreetMap: <strong>191 974</strong> POIs, <strong>11 470</strong> land-use and
  <strong>14 668</strong> public-transport features were spatially joined onto the cells.
  <strong>4436</strong> of the 4437 cells carry at least one semantic aspect and dissolve into
  <strong>4201</strong> distinct semantic contexts — <em>almost every cell is semantically
  unique</em>, with ≈15 labels each on average. <strong>100%</strong> of GPS points fall inside
  an enriched location.</p>
  <figure>
    <img src="{IMG['traj']}" alt="One trajectory over its traversed semantic cells">
    <figcaption><b>Geometric check.</b> The longest trajectory (481 points): the GPS path
    (black) and the zoom-17 cells it traverses, each coloured by its semantic context. 57 cells,
    57 distinct contexts — a direct illustration of central-Paris semantic density.</figcaption>
  </figure>
</section>

<section>
  <h2><span class="n">03</span> &nbsp;Summarization rate</h2>
  <p>With the paper's default operating point (uniform squares, resolution&nbsp;17, all aspects,
  τ&nbsp;=&nbsp;0.9) the dataset summarization rate is <strong>S_rate&nbsp;=&nbsp;0.738</strong>
  — each trajectory is reduced to ~26% as many distinct locations as it has points — ranging
  0.58–0.88 across trajectories.</p>
  <figure>
    <img src="{IMG['srate']}" alt="S_rate per trajectory bar chart">
    <figcaption><b>Per-trajectory compression</b> at τ=0.9. Mean 0.738 (dashed).
    S_rate = 1 − (representative semantic locations) / (GPS points).</figcaption>
  </figure>
</section>

<section>
  <h2><span class="n">04</span> &nbsp;Compression vs semantic quality</h2>
  <p>The similarity threshold τ trades compression against how much semantics survives,
  measured by <strong>MUITAS</strong> (Petry et&nbsp;al. 2019 — the repo's verbatim measure,
  1&nbsp;=&nbsp;semantics fully preserved). Sweeping τ reproduces the paper's core finding on
  this data:</p>
  <div class="tablewrap"><table>
    <thead><tr><th>τ</th><th>S_rate¹ <span style="font-weight:400;text-transform:none">(summ1 only)</span></th><th>S_rate² <span style="font-weight:400;text-transform:none">(MAT-SUM)</span></th><th>MUITAS</th><th></th></tr></thead>
    <tbody>
      <tr class="hi"><td>0.90</td><td>0.735</td><td>0.738</td><td>0.992</td><td><span class="pill good">sweet spot</span></td></tr>
      <tr class="hi"><td>0.80</td><td>0.735</td><td>0.797</td><td>0.821</td><td><span class="pill good">sweet spot</span></td></tr>
      <tr><td>0.70</td><td>0.735</td><td>0.893</td><td>0.353</td><td><span class="pill warn">quality drop</span></td></tr>
      <tr><td>0.60</td><td>0.735</td><td>0.948</td><td>0.154</td><td><span class="pill warn">quality drop</span></td></tr>
      <tr><td>0.50</td><td>0.735</td><td>0.974</td><td>0.060</td><td><span class="pill warn">quality drop</span></td></tr>
    </tbody>
  </table></div>
  <figure>
    <img src="{IMG['trade']}" alt="Trade-off curve of S_rate and MUITAS vs tau">
    <figcaption><b>The trade-off.</b> Compression (blue) rises and semantic quality (orange)
    falls as τ decreases. They cross near τ≈0.72; the usable region is the shaded τ≈0.8–0.9 band.</figcaption>
  </figure>
</section>

<section>
  <h2><span class="n">05</span> &nbsp;What this says about the data</h2>
  <div class="cols">
    <div class="card"><h3>Density changes the operating point</h3>
      <p style="margin:0">Because central Paris is so semantically rich, at τ=0.9 hardly any two
      locations count as "similar" (0.4 similar neighbours on average), so the similarity-merge
      step barely fires and the compression comes almost entirely from collapsing repeated
      visits to the same cell.</p></div>
    <div class="card"><h3>Lower τ recovers Geolife-like rates</h3>
      <p style="margin:0">Dropping τ to 0.6–0.7 reaches S_rate 0.89–0.95 — the range the paper
      reports for Geolife — but MUITAS shows the semantic cost is steep below 0.8. This is
      exactly the paper's EQ1 conclusion made concrete on European-city-centre data.</p></div>
  </div>
  <p style="margin-top:18px"><strong>Recommended operating point for this dataset: τ ≈ 0.8</strong>
  — S_rate 0.80 with MUITAS 0.82. If higher compression is needed, prefer a coarser tessellation
  or fewer aspects over pushing τ down, since those reduce semantic granularity more gracefully
  than the similarity threshold does here.</p>
</section>

<section>
  <h2><span class="n">06</span> &nbsp;Reproduce</h2>
  <pre class="mono">export MATSUM_PBF=/path/to/ile-de-france.osm.pbf
python matsum_prepare.py     # tessellation + OSM enrichment → semantic locations
python matsum_summarize.py   # semantic mapping + summarize → S_rate
python matsum_muitas.py      # S_rate + MUITAS vs τ
python matsum_visualize.py   # per-trajectory Folium maps
python matsum_figures.py     # static figures</pre>
  <p style="font-size:13.5px;color:var(--muted)">Full details, outputs and the 21 interactive
  maps are described in <code>matsum/README.md</code>. Parameters live at the top of each script.</p>
</section>

<footer>
  MAT-SUM · Akkodis Paris GPS · squares res17 · all aspects · generated from
  <span class="mono">matsum/</span>. Method: Pugliese, Lettich, Pinelli &amp; Renso,
  <em>Summarizing Trajectories Using Semantically Enriched Geographical Context</em>,
  SIGSPATIAL&nbsp;2023. Semantic quality: MUITAS, Petry et&nbsp;al. 2019.
</footer>
</div>
"""

out = os.path.join(HERE, "report.html")
with open(out, "w") as f:
    f.write(HTML)
print("wrote", out, f"({len(HTML)//1024} KB)")
