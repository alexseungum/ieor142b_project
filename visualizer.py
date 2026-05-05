"""
visualizer.py
Generate an interactive HTML chart visualizer using canvas rendering.

Usage:
    python visualizer.py --sm path/to/chart.sm --output chart_viz.html
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_utils import parse_sm_file, measures_to_timestep_labels


def build_chart_json(sm_path: str, difficulty_filter: str = None) -> dict:
    """Parse a .sm file and return JSON-serializable chart data."""
    sm_data = parse_sm_file(sm_path)

    chart = None
    for c in sm_data['charts']:
        if c['chart_type'].lower() in ('dance-single', 'dance single'):
            if difficulty_filter is None or c['difficulty'].lower() == difficulty_filter.lower():
                chart = c
                break

    if chart is None:
        for c in sm_data['charts']:
            if 'single' in c['chart_type'].lower():
                chart = c
                break

    if chart is None:
        raise ValueError("No dance-single chart found in .sm file")

    labels = measures_to_timestep_labels(chart['measures'], subdivision=16)
    T = len(labels)

    events = []
    for t in range(T):
        if labels[t].sum() > 0:
            arrows = [int(labels[t, i]) for i in range(4)]
            events.append({'t': t, 'arrows': arrows})

    bpm = sm_data['bpms'][0][1] if sm_data['bpms'] else 120.0

    return {
        'title': sm_data['title'],
        'bpm': bpm,
        'offset': sm_data['offset'],
        'difficulty': chart['difficulty'],
        'meter': chart['meter'],
        'total_steps': int((labels.sum(-1) > 0).sum()),
        'total_timesteps': T,
        'events': events,
    }


def build_html(chart_data: dict, audio_data_uri: str = None) -> str:
    chart_json = json.dumps(chart_data, indent=2)
    title = chart_data['title']
    diff  = chart_data['difficulty'].upper()
    meter = chart_data['meter']
    bpm   = chart_data['bpm']
    steps = chart_data['total_steps']

    # Audio element — embedded as base64 if provided, otherwise no audio
    if audio_data_uri:
        audio_tag = f'<audio id="audio" src="{audio_data_uri}" preload="auto"></audio>'
        audio_js = """
  const audio = document.getElementById('audio');
  // Sync audio with playback
  function startAudio() {
    audio.currentTime = t;
    audio.play().catch(() => {});
  }
  function pauseAudio() { audio.pause(); }
  function resetAudio() { audio.pause(); audio.currentTime = 0; }
"""
        audio_play_call   = "startAudio();"
        audio_pause_call  = "pauseAudio();"
        audio_reset_call  = "resetAudio();"
    else:
        audio_tag = ""
        audio_js  = ""
        audio_play_call  = ""
        audio_pause_call = ""
        audio_reset_call = ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — DDR Chart</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<style>
:root {{
  --red:    #ff3a3a;
  --green:  #00e676;
  --blue:   #2979ff;
  --yellow: #ffea00;
  --bg:     #080810;
  --panel:  #0e0e1a;
  --border: #1e1e3a;
  --text:   #c8c8e8;
  --dim:    #404060;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  background: var(--bg);
  color: var(--text);
  font-family: 'Share Tech Mono', monospace;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 24px 16px;
  overflow-x: hidden;
}}
body::before {{
  content:'';
  position:fixed;
  inset:0;
  background: repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.08) 2px,rgba(0,0,0,0.08) 4px);
  pointer-events:none;
  z-index:999;
}}
.header {{ text-align:center; margin-bottom:20px; }}
.title {{
  font-family:'Orbitron',sans-serif;
  font-weight:900;
  font-size:clamp(1.2em,4vw,2em);
  letter-spacing:0.05em;
  background:linear-gradient(135deg,#fff 30%,#8080ff);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
}}
.meta-row {{
  display:flex; gap:20px; justify-content:center;
  margin-top:6px; font-size:0.78em; letter-spacing:0.1em; color:var(--dim);
}}
.meta-row span {{ color:var(--text); }}
.controls {{
  display:flex; gap:10px; align-items:center;
  margin-bottom:20px; flex-wrap:wrap; justify-content:center;
}}
.btn {{
  font-family:'Orbitron',sans-serif;
  font-size:0.7em; font-weight:700; letter-spacing:0.15em;
  padding:10px 24px; border:1px solid var(--border);
  background:var(--panel); color:var(--text);
  cursor:pointer; text-transform:uppercase;
  clip-path:polygon(8px 0%,100% 0%,calc(100% - 8px) 100%,0% 100%);
  transition:all 0.15s;
}}
.btn:hover {{ background:#1a1a30; border-color:#4040aa; }}
.btn.playing {{ background:var(--red); border-color:var(--red); color:#fff; }}
.speed-wrap {{
  display:flex; align-items:center; gap:8px;
  font-size:0.75em; letter-spacing:0.1em; color:var(--dim);
}}
.speed-wrap span {{ color:var(--text); min-width:28px; }}
input[type=range] {{ width:90px; accent-color:var(--blue); cursor:pointer; }}
.main {{
  display:flex; gap:20px; align-items:flex-start;
  flex-wrap:wrap; justify-content:center;
}}
.stage-wrap {{ display:flex; flex-direction:column; align-items:center; gap:10px; }}
canvas {{ border:1px solid var(--border); display:block; }}
.progress-track {{
  width:300px; height:4px; background:var(--border); border-radius:2px; overflow:hidden;
}}
.progress-fill {{
  height:100%; width:0%;
  background:linear-gradient(90deg,var(--blue),var(--red));
  transition:width 0.05s linear;
}}
.beat-label {{ font-size:0.7em; letter-spacing:0.15em; color:var(--dim); }}
.panel {{
  width:200px; background:var(--panel);
  border:1px solid var(--border); padding:16px;
  font-size:0.75em; letter-spacing:0.08em;
}}
.panel-title {{
  font-family:'Orbitron',sans-serif; font-size:0.65em; font-weight:700;
  letter-spacing:0.2em; color:var(--dim); margin-bottom:14px; text-transform:uppercase;
}}
.panel-row {{
  display:flex; justify-content:space-between; margin-bottom:10px; color:var(--dim);
}}
.panel-row span:last-child {{
  color:var(--text); font-family:'Orbitron',sans-serif; font-size:0.9em;
}}
.divider {{ border:none; border-top:1px solid var(--border); margin:14px 0; }}
.stat-big {{
  font-family:'Orbitron',sans-serif; font-size:1.6em; font-weight:900;
  color:#fff; text-align:center; margin:4px 0 2px;
}}
.stat-label {{
  font-size:0.65em; color:var(--dim); text-align:center;
  letter-spacing:0.15em; text-transform:uppercase;
}}
.beat-dots {{ display:flex; justify-content:center; gap:6px; margin-top:14px; }}
.dot {{
  width:8px; height:8px; border-radius:50%; background:var(--border);
  transition:background 0.05s,transform 0.05s;
}}
.dot.on {{ background:#fff; transform:scale(1.4); }}
.diff-badge {{
  display:inline-block; font-family:'Orbitron',sans-serif;
  font-size:0.6em; font-weight:700; letter-spacing:0.2em;
  padding:2px 8px; border-radius:2px;
}}
.diff-BEGINNER  {{ background:#1a3a1a; color:#00e676; border:1px solid #00e676; }}
.diff-EASY      {{ background:#1a3a1a; color:#00e676; border:1px solid #00e676; }}
.diff-MEDIUM    {{ background:#3a2a00; color:#ffea00; border:1px solid #ffea00; }}
.diff-HARD      {{ background:#3a1a00; color:#ff9100; border:1px solid #ff9100; }}
.diff-CHALLENGE {{ background:#3a0000; color:#ff3a3a; border:1px solid #ff3a3a; }}
</style>
</head>
<body>
{audio_tag}

<div class="header">
  <div class="title">{title}</div>
  <div class="meta-row">
    <span class="diff-badge diff-{diff}">{diff}</span>
    <span>METER <span>{meter}</span></span>
    <span>BPM <span>{bpm:.0f}</span></span>
    <span>STEPS <span>{steps}</span></span>
  </div>
</div>

<div class="controls">
  <button class="btn" id="btn-play" onclick="togglePlay()">▶ PLAY</button>
  <button class="btn" onclick="resetViz()">↺ RESET</button>
  <div class="speed-wrap">
    SPEED
    <input type="range" id="speed-slider" min="0.5" max="4" step="0.25" value="2"
           oninput="speed=parseFloat(this.value);document.getElementById('speed-val').textContent=this.value+'×'">
    <span id="speed-val">2×</span>
  </div>
</div>

<div class="main">
  <div class="stage-wrap">
    <canvas id="c" width="300" height="520"></canvas>
    <div class="progress-track"><div class="progress-fill" id="prog"></div></div>
    <div class="beat-label" id="beat-lbl">BEAT 0</div>
  </div>
  <div class="panel">
    <div class="panel-title">Chart Info</div>
    <div class="panel-row"><span>Title</span><span>{title[:14]}</span></div>
    <div class="panel-row"><span>BPM</span><span>{bpm:.0f}</span></div>
    <div class="panel-row"><span>Steps</span><span>{steps}</span></div>
    <div class="panel-row"><span>Diff</span><span>{diff}</span></div>
    <hr class="divider">
    <div class="panel-title">Live</div>
    <div class="stat-big" id="stat-combo">0</div>
    <div class="stat-label">COMBO</div>
    <div style="margin-top:10px" class="stat-big" id="stat-hits">0</div>
    <div class="stat-label">STEPS HIT</div>
    <div class="beat-dots">
      <div class="dot" id="d0"></div><div class="dot" id="d1"></div>
      <div class="dot" id="d2"></div><div class="dot" id="d3"></div>
    </div>
  </div>
</div>

<script>
const CHART = {chart_json};
{audio_js}

const W = 300, H = 520;
const COL_XS = [14, 79, 144, 209];
const ARROW_SIZE = 52;
const HIT_Y = H - 72;
const COL_COLORS = ['#ff3a3a','#00e676','#2979ff','#ffea00'];
const COL_ARROWS = ['←','↓','↑','→'];
const BPM = CHART.bpm;
const SEC_PER_SUBDIV = (60 / BPM) / 4;
const TOTAL_DUR = CHART.total_timesteps * SEC_PER_SUBDIV;

const events = CHART.events.map(e => ({{...e, t_sec: e.t * SEC_PER_SUBDIV}}));

let playing = false, t = 0, lastTs = null, speed = 2.0;
let combo = 0, hits = 0;
let firedSet = new Set();
let recFlash = [0,0,0,0];

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');

function pps() {{ return (HIT_Y / 1.8) * speed; }}
function eventY(t_sec) {{ return HIT_Y - (t_sec - t) * pps(); }}

function drawRoundRect(x, y, w, h, r, fill, stroke, lw) {{
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
  if (fill)   {{ ctx.fillStyle = fill; ctx.fill(); }}
  if (stroke) {{ ctx.strokeStyle = stroke; ctx.lineWidth = lw || 2; ctx.stroke(); }}
}}

function drawArrow(col, y, alpha) {{
  if (alpha <= 0) return;
  const x = COL_XS[col], sz = ARROW_SIZE, color = COL_COLORS[col];
  ctx.globalAlpha = Math.min(1, alpha);
  ctx.shadowColor = color; ctx.shadowBlur = 14;
  drawRoundRect(x, y - sz/2, sz, sz, 8, color + '28', color, 2);
  ctx.shadowBlur = 0;
  ctx.fillStyle = color;
  ctx.font = `bold ${{Math.round(sz * 0.62)}}px monospace`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(COL_ARROWS[col], x + sz/2, y);
  ctx.globalAlpha = 1;
}}

function drawReceptor(col) {{
  const x = COL_XS[col], sz = ARROW_SIZE, color = COL_COLORS[col];
  const flash = recFlash[col] > 0;
  ctx.globalAlpha = flash ? 1.0 : 0.18;
  if (flash) {{ ctx.shadowColor = color; ctx.shadowBlur = 24; }}
  drawRoundRect(x, HIT_Y - sz/2, sz, sz, 8, flash ? color + '55' : 'transparent', color, 2);
  ctx.shadowBlur = 0;
  ctx.fillStyle = color;
  ctx.font = `bold ${{Math.round(sz * 0.62)}}px monospace`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(COL_ARROWS[col], x + sz/2, HIT_Y);
  ctx.globalAlpha = 1;
}}

function draw() {{
  ctx.fillStyle = '#080810';
  ctx.fillRect(0, 0, W, H);

  // lane lines
  ctx.strokeStyle = '#1e1e3a'; ctx.lineWidth = 1;
  for (let c = 0; c <= 4; c++) {{
    const lx = 14 + c * 65 - 1;
    ctx.beginPath(); ctx.moveTo(lx, 0); ctx.lineTo(lx, H); ctx.stroke();
  }}

  // beat grid
  const beatSec = 60 / BPM;
  const startBeat = Math.floor((t - HIT_Y / pps()) / beatSec) * beatSec;
  for (let b = startBeat; b < t + H / pps() + beatSec; b += beatSec) {{
    const gy = eventY(b);
    if (gy < -2 || gy > H + 2) continue;
    const isMeasure = Math.round(b / beatSec) % 4 === 0;
    ctx.strokeStyle = isMeasure ? '#2a2a50' : '#111128';
    ctx.lineWidth = isMeasure ? 1.5 : 0.5;
    ctx.beginPath(); ctx.moveTo(14, gy); ctx.lineTo(W - 14, gy); ctx.stroke();
  }}

  // hit line
  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, HIT_Y); ctx.lineTo(W, HIT_Y); ctx.stroke();

  // receptors
  for (let c = 0; c < 4; c++) drawReceptor(c);

  // arrows — only those in visible window
  const visTop    = t - ARROW_SIZE / pps();
  const visBottom = t + (H + ARROW_SIZE * 2) / pps();

  for (const ev of events) {{
    if (ev.t_sec < visTop)    continue;
    if (ev.t_sec > visBottom) break;
    const y = eventY(ev.t_sec);

    // fire flash when crossing hit zone
    if (!firedSet.has(ev.t) && y >= HIT_Y - 8 && y <= HIT_Y + 32) {{
      firedSet.add(ev.t);
      for (let c = 0; c < 4; c++) if (ev.arrows[c]) recFlash[c] = 6;
      combo++; hits++;
      document.getElementById('stat-combo').textContent = combo;
      document.getElementById('stat-hits').textContent  = hits;
    }}

    // fade out after hit zone
    const alpha = y > HIT_Y + 10
      ? Math.max(0, 1 - (y - HIT_Y - 10) / 50)
      : 1.0;

    for (let c = 0; c < 4; c++) if (ev.arrows[c]) drawArrow(c, y, alpha);
  }}

  for (let c = 0; c < 4; c++) if (recFlash[c] > 0) recFlash[c]--;
}}

function frame(ts) {{
  if (!playing) return;
  if (lastTs !== null) t += (ts - lastTs) / 1000;
  lastTs = ts;
  if (t > TOTAL_DUR + 2) {{ resetViz(); return; }}
  draw();
  document.getElementById('prog').style.width =
    (Math.min(1, t / TOTAL_DUR) * 100).toFixed(2) + '%';
  const beat = Math.floor(t / (60 / BPM));
  document.getElementById('beat-lbl').textContent = 'BEAT ' + beat;
  const b4 = beat % 4;
  for (let i = 0; i < 4; i++)
    document.getElementById('d' + i).classList.toggle('on', i === b4);
  requestAnimationFrame(frame);
}}

function togglePlay() {{
  playing = !playing;
  const btn = document.getElementById('btn-play');
  if (playing) {{
    btn.textContent = '⏸ PAUSE'; btn.classList.add('playing');
    lastTs = null;
    {audio_play_call}
    requestAnimationFrame(frame);
  }} else {{
    btn.textContent = '▶ PLAY'; btn.classList.remove('playing');
    {audio_pause_call}
  }}
}}

function resetViz() {{
  playing = false; t = 0; combo = 0; hits = 0;
  firedSet.clear(); recFlash = [0,0,0,0];
  {audio_reset_call}
  document.getElementById('btn-play').textContent = '▶ PLAY';
  document.getElementById('btn-play').classList.remove('playing');
  document.getElementById('stat-combo').textContent = '0';
  document.getElementById('stat-hits').textContent  = '0';
  document.getElementById('prog').style.width = '0%';
  document.getElementById('beat-lbl').textContent = 'BEAT 0';
  for (let i = 0; i < 4; i++) document.getElementById('d'+i).classList.remove('on');
  draw();
}}

draw();
</script>
</body>
</html>"""


def main():
    p = argparse.ArgumentParser(description='DDR Chart Visualizer')
    p.add_argument('--sm',         type=str, required=True, help='Path to .sm file')
    p.add_argument('--difficulty', type=str, default=None,  help='Difficulty to visualize (optional)')
    p.add_argument('--output',     type=str, default='chart_viz.html', help='Output HTML path')
    args = p.parse_args()

    print(f"Parsing {args.sm}...")
    chart_data = build_chart_json(args.sm, args.difficulty)
    print(f"  Title: {chart_data['title']}")
    print(f"  Difficulty: {chart_data['difficulty']}  Meter: {chart_data['meter']}")
    print(f"  Total steps: {chart_data['total_steps']}  BPM: {chart_data['bpm']:.1f}")

    html = build_html(chart_data)
    with open(args.output, 'w') as f:
        f.write(html)
    print(f"Visualizer written to: {args.output}")
    print(f"Open {args.output} in your browser!")


if __name__ == '__main__':
    main()