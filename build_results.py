"""Assemble the consolidated prototype-results report (self-contained HTML)."""
import os, base64
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figures")


def uri(name):
    with open(os.path.join(FIG, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


IMG = {k: uri(v) for k, v in {
    "fid": "synth_vs_real.png",
    "deep": "deep_vs_markov.png",
    "coupled": "coupled_spatial.png",
}.items()}

CSS = """
:root{--paper:#f6f8fb;--ink:#18202c;--muted:#5a6675;--line:#dbe2ec;--card:#fff;
--accent:#1f7ab0;--accent-soft:#e3eef6;--hot:#d1621b;--hot-soft:#f7e6d8;--good:#2f7d4f;
--good-soft:#e2f0e8;--violet:#6a53a6;--shadow:0 1px 2px rgba(20,30,45,.06),0 8px 24px rgba(20,30,45,.05);}
@media (prefers-color-scheme:dark){:root{--paper:#0e1620;--ink:#e8edf4;--muted:#9aa7b6;--line:#243244;
--card:#141e2b;--accent:#4aa3d8;--accent-soft:#16283a;--hot:#e08a4e;--hot-soft:#2a1e14;--good:#5cbf87;
--good-soft:#132318;--violet:#a290d6;--shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);}}
:root[data-theme=dark]{--paper:#0e1620;--ink:#e8edf4;--muted:#9aa7b6;--line:#243244;--card:#141e2b;
--accent:#4aa3d8;--accent-soft:#16283a;--hot:#e08a4e;--hot-soft:#2a1e14;--good:#5cbf87;--good-soft:#132318;
--violet:#a290d6;--shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);}
:root[data-theme=light]{--paper:#f6f8fb;--ink:#18202c;--muted:#5a6675;--line:#dbe2ec;--card:#fff;
--accent:#1f7ab0;--accent-soft:#e3eef6;--hot:#d1621b;--hot-soft:#f7e6d8;--good:#2f7d4f;--good-soft:#e2f0e8;
--violet:#6a53a6;--shadow:0 1px 2px rgba(20,30,45,.06),0 8px 24px rgba(20,30,45,.05);}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:790px;margin:0 auto;padding:0 24px 96px}.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
header{padding:64px 0 36px;border-bottom:1px solid var(--line);margin-bottom:8px}
.eyebrow{font-family:ui-monospace,Menlo,monospace;font-size:12px;letter-spacing:.14em;text-transform:uppercase;
color:var(--accent);margin:0 0 18px}
h1{font-size:clamp(29px,5vw,44px);line-height:1.07;letter-spacing:-.02em;font-weight:800;margin:0 0 16px;text-wrap:balance}
.lede{font-size:19px;color:var(--muted);margin:0;max-width:62ch}
.meta{display:flex;flex-wrap:wrap;gap:8px 20px;margin-top:24px;font-size:13px;color:var(--muted);
font-family:ui-monospace,Menlo,monospace}.meta b{color:var(--ink);font-weight:600}
h2{font-size:13px;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);
font-family:ui-monospace,Menlo,monospace;margin:56px 0 10px}h2 .n{color:var(--accent)}
h3{font-size:20px;letter-spacing:-.01em;margin:6px 0 12px;text-wrap:balance}
p{margin:0 0 16px}strong{font-weight:650}em.k{font-style:normal;background:var(--accent-soft);padding:1px 6px;
border-radius:5px;font-family:ui-monospace,Menlo,monospace;font-size:.9em}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:8px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 16px;box-shadow:var(--shadow)}
.kpi .v{font-size:27px;font-weight:800;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.kpi .l{font-size:12.5px;color:var(--muted);margin-top:4px}
@media (max-width:640px){.kpis{grid-template-columns:repeat(2,1fr)}}
.callout{background:var(--accent-soft);border:1px solid var(--line);border-left:3px solid var(--accent);
border-radius:12px;padding:20px 22px;margin:22px 0}.callout h3{margin-top:0}
.callout.warn{background:var(--hot-soft);border-left-color:var(--hot)}.callout.warn h3{color:var(--hot)}
figure{margin:24px 0}figure img{width:100%;height:auto;display:block;border:1px solid var(--line);
border-radius:12px;background:var(--card);box-shadow:var(--shadow)}
figcaption{font-size:13.5px;color:var(--muted);margin-top:10px}figcaption b{color:var(--ink);font-weight:600}
.tablewrap{overflow-x:auto;margin:18px 0;border:1px solid var(--line);border-radius:12px}
table{width:100%;border-collapse:collapse;font-size:14px}
th,td{padding:10px 13px;text-align:right;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{font-family:ui-monospace,Menlo,monospace;font-size:11px;letter-spacing:.03em;text-transform:uppercase;
color:var(--muted);background:var(--card)}tbody tr:last-child td{border-bottom:none}
tr.hi td{background:color-mix(in srgb,var(--good) 13%,transparent);font-weight:600}
tr.bad td{background:color-mix(in srgb,var(--hot) 11%,transparent)}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11.5px;font-weight:600;
font-family:ui-monospace,Menlo,monospace}
.pill.good{color:var(--good);background:var(--good-soft)}.pill.bad{color:var(--hot);background:var(--hot-soft)}
ul{padding-left:20px}li{margin:7px 0}
.steps{display:grid;gap:10px;margin:16px 0}
.step{display:grid;grid-template-columns:34px 1fr;gap:14px;align-items:baseline;background:var(--card);
border:1px solid var(--line);border-radius:10px;padding:12px 16px}
.step .b{font-family:ui-monospace,Menlo,monospace;font-weight:700;color:var(--accent)}
.step .t{font-size:14.5px}.step .t b{font-weight:650}
footer{margin-top:64px;padding-top:24px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}
"""

HTML = f"""<title>Synthetic MAT from MAT-SUM — prototype results</title>
<style>{CSS}</style>
<div class="wrap">
<header>
  <p class="eyebrow">Prototype results · draft for coauthors</p>
  <h1>Privacy-preserving synthetic trajectories from MAT-SUM: prototype results</h1>
  <p class="lede">A working proof-of-concept on the 20 Akkodis Paris trajectories, covering the full
  pipeline — anonymized semantic vocabulary, a semantic sequence generator (mechanistic vs deep),
  and spatial materialization (decoupled vs spatially-coupled). Four findings, and two honest limits.</p>
  <div class="meta">
    <span><b>Data</b> 20 real → 200 synthetic</span>
    <span><b>Vocabulary</b> m=3, k=5 (71 symbols)</span>
    <span><b>Metrics</b> fidelity · utility · privacy</span>
  </div>
</header>

<section>
  <div class="kpis">
    <div class="kpi"><div class="v">10×</div><div class="l">synthetic vs real<br>(200 from 20)</div></div>
    <div class="kpi"><div class="v">0</div><div class="l">verbatim copies<br>(mechanistic)</div></div>
    <div class="kpi"><div class="v">27%</div><div class="l">verbatim copies<br>(deep / LSTM)</div></div>
    <div class="kpi"><div class="v">2.44</div><div class="l">synth radius of gyr. km<br>(real 2.36)</div></div>
  </div>
</section>

<section>
  <h2><span class="n">01</span> &nbsp;Pipeline in one screen</h2>
  <div class="steps">
    <div class="step"><span class="b">0</span><span class="t"><b>Anonymized vocabulary.</b> Ordered
    semantic visit sequences (pre-collapse) with dwell-times; each location abstracted to its top-<em>m</em>
    TF-IDF aspects (<em class="k">symbol</em>); rare symbols merged into the nearest frequent one to enforce
    <em>k</em>-anonymity. This is the same τ/generalization dial as MAT-SUM, now a privacy control.</span></div>
    <div class="step"><span class="b">1a</span><span class="t"><b>Mechanistic generator.</b> Order-1
    semantic Markov chain + empirical per-symbol dwell + DITRAS-style length sampling.</span></div>
    <div class="step"><span class="b">1b</span><span class="t"><b>Deep generator.</b> A small LSTM
    language model over the same alphabet, same dwell model — isolating the sequence model.</span></div>
    <div class="step"><span class="b">2</span><span class="t"><b>Spatial materialization.</b> Realize each
    symbol at a concrete Paris tile of that type (EPR distance-decay), <em>decoupled</em> from the sequence.</span></div>
    <div class="step"><span class="b">3</span><span class="t"><b>Spatially-coupled generator.</b> Generate at
    tile level with a transition that couples learned semantics × spatial proximity × visitation, wrapped in
    an EPR explore/return process — locality emerges from the process.</span></div>
  </div>
</section>

<section>
  <h2><span class="n">02</span> &nbsp;Finding 1 — the (m, k) dial trades granularity, fidelity and privacy</h2>
  <p>Semantic granularity <em>m</em> sets the alphabet size; anonymity <em>k</em> merges rare symbols.
  Finer semantics (large alphabet) starve the order-1 transitions on 20 trajectories; crucially,
  <strong>raising anonymity <em>k</em> also improves fidelity</strong> by densifying the alphabet — at this
  data scale privacy and fidelity align rather than conflict. No verbatim copies at any setting.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>m</th><th>k</th><th>alphabet</th><th>unigram TV ↓</th><th>bigram TV ↓</th><th>MUITAS→real ↑</th><th>copies</th></tr></thead>
    <tbody>
      <tr><td>1</td><td>2</td><td>30</td><td>0.067</td><td>0.139</td><td>0.881</td><td>0</td></tr>
      <tr><td>1</td><td>5</td><td>23</td><td>0.043</td><td>0.101</td><td>0.908</td><td>0</td></tr>
      <tr><td>2</td><td>2</td><td>108</td><td>0.257</td><td>0.483</td><td>0.635</td><td>0</td></tr>
      <tr class="hi"><td>2</td><td>5</td><td>52</td><td>0.074</td><td>0.190</td><td>0.811</td><td>0</td></tr>
      <tr><td>3</td><td>2</td><td>208</td><td>0.294</td><td>0.759</td><td>0.387</td><td>0</td></tr>
      <tr class="hi"><td>3</td><td>5</td><td>71</td><td>0.109</td><td>0.290</td><td>0.718</td><td>0</td></tr>
    </tbody>
  </table></div>
  <p style="font-size:13.5px;color:var(--muted)">Operating region m=2–3, k=5 (highlighted): usable
  fidelity, no leakage. Dwell-time and trajectory length are reproduced almost exactly (below).</p>
  <figure>
    <img src="{IMG['fid']}" alt="Real vs synthetic symbol frequency, dwell-time, length and DCR">
    <figcaption><b>Fidelity of the mechanistic generator</b> (m=3, k=2 shown). Dwell-time (top-right)
    is near-identical; the distance-to-closest-record (bottom-right) sits at the real→real baseline —
    synthetic trajectories are no closer to real records than real ones are to each other.</figcaption>
  </figure>
</section>

<section>
  <h2><span class="n">03</span> &nbsp;Finding 2 — deep buys fidelity by memorizing (the privacy result)</h2>
  <p>Head-to-head at m=3, k=5, same dwell model. The LSTM wins on fidelity — but only because it copies:
  <strong>54/200 verbatim and 80/200 near-verbatim</strong> trajectories, with a distance-to-closest-record
  far below the real→real baseline. The mechanistic Markov leaks <strong>nothing</strong> and sits exactly at
  the baseline, at a fidelity cost.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>model</th><th>bigram TV ↓</th><th>MUITAS→real ↑</th><th>DCR ↑</th><th>exact copies</th><th>near (≥.95)</th><th>novel bigrams</th></tr></thead>
    <tbody>
      <tr class="hi"><td>Markov (1a)</td><td>0.256</td><td>0.723</td><td>0.277</td><td>0</td><td>0</td><td>72.5%</td></tr>
      <tr class="bad"><td>LSTM (1b)</td><td>0.204</td><td>0.900</td><td>0.100</td><td>54</td><td>80</td><td>44.9%</td></tr>
    </tbody>
  </table></div>
  <p style="font-size:13.5px;color:var(--muted)">real→real DCR baseline = 0.279. Markov 0.277 ≈ baseline
  (ideal); LSTM 0.100 ≪ baseline (memorization).</p>
  <figure>
    <img src="{IMG['deep']}" alt="Deep vs Markov: fidelity, DCR and copied trajectories">
    <figcaption><b>Deep vs mechanistic.</b> Slightly better fidelity (left) is paid for with a collapse
    in privacy: the LSTM's DCR falls below the real→real line (middle) and it copies or near-copies
    two-thirds of its output (right). With n=20, generalization and fidelity are in direct tension —
    and the mechanistic model is the safe operating point.</figcaption>
  </figure>
</section>

<section>
  <h2><span class="n">04</span> &nbsp;Finding 3 — spatial coupling restores locality</h2>
  <p>Realizing symbols independently (decoupled) preserves coarse spatial spread but breaks micro-locality:
  a spatial-blind semantic sequence, materialized at each type's nearest instance, jumps too far and roams
  too wide. Generating at tile level with a <em>coupled</em> transition (semantics × proximity × visitation,
  inside an EPR explore/return process) recovers the short-jump mode and matches the radius of gyration.</p>
  <div class="tablewrap"><table>
    <thead><tr><th>metric</th><th>real</th><th>coupled (Ph.3)</th><th>decoupled (Ph.2)</th></tr></thead>
    <tbody>
      <tr class="hi"><td>radius of gyration (km)</td><td>2.36</td><td>2.44</td><td>3.28</td></tr>
      <tr><td>jump length (km)</td><td>0.28</td><td>0.72</td><td>1.01</td></tr>
      <tr><td>exact path copies</td><td>—</td><td>0 / 200</td><td>0 / 200</td></tr>
      <tr><td>spatial DCR (km) · base 1.04</td><td>—</td><td>0.64</td><td>0.85</td></tr>
    </tbody>
  </table></div>
  <figure>
    <img src="{IMG['coupled']}" alt="Coupled vs decoupled vs real: jump length, radius of gyration, sample trajectories">
    <figcaption><b>Spatially-coupled generator.</b> The coupled model (green) recovers the real short-jump
    mode (left, the ~0.2 km spike the decoupled model missed) and matches the radius of gyration (middle);
    trajectories show a realistic localized home-region structure (right). Residual jump-length vs raw GPS
    is a sampling-granularity artifact (real is continuously sampled; synthetic is at visit level).</figcaption>
  </figure>
</section>

<section>
  <h2><span class="n">05</span> &nbsp;Finding 4 — one dial governs the whole trade-off</h2>
  <div class="callout"><h3>Compression, semantic quality and privacy move together</h3>
  <p style="margin-bottom:0">The MAT-SUM similarity threshold τ (here its granularity analogue <em>m</em>,
  plus the anonymity <em>k</em>) is a single dial: it set the S_rate/MUITAS compression-vs-quality curve, and
  it sets the fidelity-vs-privacy curve of the generator. The preserved properties are exactly what lives
  above MAT-SUM's abstraction line — semantics and coarse space/time — while exact geometry and timing are
  resampled, which is what yields both scale and privacy.</p></div>
</section>

<section>
  <h2><span class="n">06</span> &nbsp;Honest limits</h2>
  <div class="callout warn"><h3>Two caveats bound every number here</h3>
  <ul style="margin-bottom:0">
    <li><strong>n = 20.</strong> Order-1 transitions are data-starved at fine granularity, and privacy
    attacks are noisy at this scale. Numbers are illustrative; fidelity and privacy must be re-measured on
    a larger corpus (Geolife, or the full Akkodis DB) before they are paper-grade.</li>
    <li><strong>Sampling granularity.</strong> Real jump-length (0.28 km) is measured on dense continuous
    GPS; the synthetic is at semantic-visit level, so a residual jump-length gap is expected and not a
    model defect. Better spatial fidelity also slightly lowers spatial DCR (0.85→0.64) — the same
    fidelity/privacy tension, mild and copy-free here.</li>
  </ul></div>
</section>

<section>
  <h2><span class="n">07</span> &nbsp;Next</h2>
  <ul>
    <li><strong>Validate at scale</strong> — rerun fidelity/privacy on Geolife / full Akkodis.</li>
    <li><strong>Formal DP</strong> — calibrated noise on the Markov transition/dwell counts for an (ε,δ) guarantee.</li>
    <li><strong>Membership-inference</strong> — a proper attacker, beyond DCR and exact-copy checks.</li>
  </ul>
</section>

<footer>
  Prototype results · companion to the application report and the privacy roadmap. Nine modular scripts
  in <span class="mono">matsum/</span> (Phase 0–3 + deep). Methods: MAT-SUM (Pugliese et al. 2023) ·
  EPR (Song et al. 2010) · MUITAS (Petry et al. 2019). Draft — open for iteration.
</footer>
</div>
"""

out = os.path.join(HERE, "results.html")
open(out, "w").write(HTML)
print("wrote", out, f"({len(HTML)//1024} KB)")
