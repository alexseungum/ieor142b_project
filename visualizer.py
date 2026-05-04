"""
visualizer.py
Generate an interactive HTML chart visualizer.
Given a .sm file (or raw arrays), produces a standalone HTML page
that plays the chart scrolling upward in sync with the song BPM.

Usage:
    python visualizer.py --sm path/to/chart.sm --bpm 140 --output chart_viz.html
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_utils import parse_sm_file, measures_to_timestep_labels


ARROW_LABELS = ['←', '↓', '↑', '→']
ARROW_COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12']  # L D U R


def build_chart_json(sm_path: str, difficulty_filter: str = None) -> dict:
    """Parse .sm file and return JSON-serializable chart data."""
    sm_data = parse_sm_file(sm_path)

    chart = None
    for c in sm_data['charts']:
        if c['chart_type'].lower() in ('dance-single', 'dance single'):
            if difficulty_filter is None or c['difficulty'].lower() == difficulty_filter.lower():
                chart = c
                break

    if chart is None:
        # Just use the first single chart
        for c in sm_data['charts']:
            if 'single' in c['chart_type'].lower():
                chart = c
                break

    if chart is None:
        raise ValueError("No dance-single chart found in .sm file")

    labels = measures_to_timestep_labels(chart['measures'], subdivision=16)  # (T, 4)
    T = len(labels)

    # Build step events list
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


def build_html(chart_data: dict) -> str:
    chart_json = json.dumps(chart_data, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DDR Chart Visualizer — {chart_data['title']}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0d0d0d;
    color: #fff;
    font-family: 'Courier New', monospace;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
    padding: 20px;
  }}
  h1 {{ font-size: 1.4em; color: #fff; margin-bottom: 4px; text-align: center; }}
  .meta {{ color: #888; font-size: 0.85em; margin-bottom: 16px; text-align: center; }}
  .controls {{
    display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; justify-content: center;
  }}
  button {{
    padding: 8px 20px;
    background: #222;
    color: #fff;
    border: 1px solid #444;
    border-radius: 6px;
    cursor: pointer;
    font-family: inherit;
    font-size: 0.9em;
    transition: background 0.2s;
  }}
  button:hover {{ background: #333; }}
  button.active {{ background: #e74c3c; border-color: #e74c3c; }}
  .speed-control {{
    display: flex; align-items: center; gap: 8px; color: #aaa; font-size: 0.85em;
  }}
  input[type=range] {{ width: 100px; accent-color: #e74c3c; }}

  /* Main play area */
  .stage-wrapper {{
    display: flex;
    gap: 32px;
    align-items: flex-start;
  }}
  .stage {{
    position: relative;
    width: 320px;
    height: 520px;
    background: #111;
    border: 1px solid #333;
    border-radius: 8px;
    overflow: hidden;
  }}

  /* Hit zone (bottom) */
  .hit-zone {{
    position: absolute;
    bottom: 60px;
    left: 0; right: 0;
    height: 3px;
    background: rgba(255,255,255,0.08);
    z-index: 5;
  }}

  /* Receptor arrows (stationary targets) */
  .receptors {{
    position: absolute;
    bottom: 44px;
    left: 0; right: 0;
    display: flex;
    justify-content: space-around;
    padding: 0 16px;
    z-index: 10;
  }}
  .receptor {{
    width: 56px; height: 56px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 2em;
    opacity: 0.25;
    border: 2px solid currentColor;
    transition: opacity 0.05s;
  }}
  .receptor.hit {{ opacity: 1; transform: scale(1.1); transition: transform 0.05s; }}

  /* Scrolling arrow container */
  #arrow-canvas {{
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
  }}

  /* Individual arrows (DOM approach for simplicity) */
  .arrow {{
    position: absolute;
    width: 54px; height: 54px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.8em;
    font-weight: bold;
    border: 2px solid rgba(255,255,255,0.3);
    pointer-events: none;
    transition: opacity 0.1s;
  }}

  /* Side panel: stats */
  .stats-panel {{
    width: 200px;
    background: #111;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 16px;
    font-size: 0.85em;
  }}
  .stats-panel h3 {{ color: #aaa; margin-bottom: 12px; font-size: 0.9em; letter-spacing: 1px; }}
  .stat-row {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    color: #ccc;
  }}
  .stat-val {{ color: #fff; font-weight: bold; }}
  .difficulty-bar {{
    height: 8px;
    background: #222;
    border-radius: 4px;
    overflow: hidden;
    margin-top: 4px;
    margin-bottom: 12px;
  }}
  .difficulty-fill {{
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #2ecc71, #f39c12, #e74c3c);
  }}
  .beat-indicator {{
    margin-top: 12px;
    text-align: center;
  }}
  .beat-dot {{
    display: inline-block;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: #333;
    margin: 2px;
    transition: background 0.05s;
  }}
  .beat-dot.on {{ background: #fff; }}

  /* Progress bar */
  .progress-bar {{
    width: 320px;
    height: 6px;
    background: #222;
    border-radius: 3px;
    margin-top: 12px;
    overflow: hidden;
  }}
  .progress-fill {{
    height: 100%;
    background: linear-gradient(90deg, #e74c3c, #f39c12);
    border-radius: 3px;
    transition: width 0.05s linear;
  }}
  #beat-counter {{ font-size: 1em; color: #888; margin-top: 8px; }}
</style>
</head>
<body>

<h1>🎮 {chart_data['title']}</h1>
<div class="meta">
  {chart_data['difficulty'].upper()} ★ Meter {chart_data['meter']} &nbsp;|&nbsp;
  {chart_data['total_steps']} steps &nbsp;|&nbsp;
  {chart_data['bpm']:.1f} BPM
</div>

<div class="controls">
  <button id="btn-play" onclick="togglePlay()">▶ Play</button>
  <button onclick="resetChart()">↺ Reset</button>
  <div class="speed-control">
    Speed:
    <input type="range" id="speed-slider" min="0.5" max="3" step="0.25" value="1.5"
           oninput="updateSpeed(this.value)">
    <span id="speed-label">1.5×</span>
  </div>
</div>

<div class="stage-wrapper">
  <div>
    <div class="stage" id="stage">
      <div class="hit-zone"></div>
      <div class="receptors" id="receptors">
        <div class="receptor" id="rec-0" style="color:#e74c3c">←</div>
        <div class="receptor" id="rec-1" style="color:#2ecc71">↓</div>
        <div class="receptor" id="rec-2" style="color:#3498db">↑</div>
        <div class="receptor" id="rec-3" style="color:#f39c12">→</div>
      </div>
      <div id="arrow-canvas"></div>
    </div>
    <div class="progress-bar">
      <div class="progress-fill" id="progress-fill" style="width:0%"></div>
    </div>
    <div id="beat-counter" style="text-align:center; margin-top:6px;">Beat: 0</div>
  </div>

  <div class="stats-panel">
    <h3>CHART INFO</h3>
    <div class="stat-row"><span>Title</span><span class="stat-val" style="font-size:0.8em">{chart_data['title'][:16]}</span></div>
    <div class="stat-row"><span>BPM</span><span class="stat-val">{chart_data['bpm']:.0f}</span></div>
    <div class="stat-row"><span>Steps</span><span class="stat-val">{chart_data['total_steps']}</span></div>
    <div class="stat-row"><span>Difficulty</span><span class="stat-val">{chart_data['difficulty'].upper()}</span></div>
    <div class="difficulty-bar">
      <div class="difficulty-fill" style="width:{min(100, chart_data['meter']*10)}%"></div>
    </div>

    <h3>LIVE</h3>
    <div class="stat-row"><span>Combo</span><span class="stat-val" id="stat-combo">0</span></div>
    <div class="stat-row"><span>Steps hit</span><span class="stat-val" id="stat-hits">0</span></div>

    <div class="beat-indicator">
      <div class="beat-dot" id="bd-0"></div>
      <div class="beat-dot" id="bd-1"></div>
      <div class="beat-dot" id="bd-2"></div>
      <div class="beat-dot" id="bd-3"></div>
    </div>
  </div>
</div>

<script>
const CHART = {chart_json};

const COLS = 4;
const STAGE_W = 320;
const STAGE_H = 520;
const HIT_Y = STAGE_H - 60 - 28;  // center of receptor zone (from top)
const COL_COLORS = ['#e74c3c','#2ecc71','#3498db','#f39c12'];
const COL_ARROWS = ['←','↓','↑','→'];
const COL_XS = [16 + 0*72, 16 + 1*72, 16 + 2*72, 16 + 3*72];  // x positions

let playing = false;
let currentTime = 0;  // in seconds
let lastTimestamp = null;
let speedMultiplier = 1.5;  // pixels per 16th-note subdivision

// Build events with time in seconds
const BPM = CHART.bpm;
const SUBDIVISION = 16;  // 16 rows per measure = 16th notes
const secPerSubdiv = (60 / BPM) / 4;  // time between 16th notes

const events = CHART.events.map(e => ({{
  ...e,
  timeSec: e.t * secPerSubdiv - CHART.offset,
}}));

// Visual state
let combo = 0;
let hitsShown = 0;
let activeArrows = [];  // DOM elements currently on screen

const canvas = document.getElementById('arrow-canvas');
const stage = document.getElementById('stage');

// Scroll speed: pixels per second (arrows travel from top to hit zone)
function getScrollPPS() {{
  // Base: at 1x speed, an arrow takes 2 seconds to travel the visible area
  return (HIT_Y / 2.0) * speedMultiplier;
}}

// Convert event time → y position at current playback time
function eventToY(eventTimeSec) {{
  const pps = getScrollPPS();
  const dt = eventTimeSec - currentTime;  // positive = future
  return HIT_Y - dt * pps;
}}

function updateSpeed(val) {{
  speedMultiplier = parseFloat(val);
  document.getElementById('speed-label').textContent = val + '×';
}}

function togglePlay() {{
  playing = !playing;
  const btn = document.getElementById('btn-play');
  if (playing) {{
    btn.textContent = '⏸ Pause';
    btn.classList.add('active');
    lastTimestamp = null;
    requestAnimationFrame(frame);
  }} else {{
    btn.textContent = '▶ Play';
    btn.classList.remove('active');
  }}
}}

function resetChart() {{
  playing = false;
  currentTime = 0;
  combo = 0; hitsShown = 0;
  document.getElementById('btn-play').textContent = '▶ Play';
  document.getElementById('btn-play').classList.remove('active');
  // Clear arrows
  canvas.innerHTML = '';
  activeArrows = [];
  updateStats();
  document.getElementById('progress-fill').style.width = '0%';
  document.getElementById('beat-counter').textContent = 'Beat: 0';
  // Reset receptors
  for (let i = 0; i < 4; i++) {{
    document.getElementById('rec-' + i).classList.remove('hit');
  }}
}}

function updateStats() {{
  document.getElementById('stat-combo').textContent = combo;
  document.getElementById('stat-hits').textContent = hitsShown;
}}

function flashReceptor(col) {{
  const rec = document.getElementById('rec-' + col);
  rec.classList.add('hit');
  setTimeout(() => rec.classList.remove('hit'), 80);
}}

function beatPulse(beat) {{
  const idx = Math.floor(beat) % 4;
  for (let i = 0; i < 4; i++) {{
    document.getElementById('bd-' + i).classList.toggle('on', i === idx);
  }}
}}

let lastRenderedEvent = -1;

function frame(timestamp) {{
  if (!playing) return;
  if (lastTimestamp !== null) {{
    const dt = (timestamp - lastTimestamp) / 1000;
    currentTime += dt;
  }}
  lastTimestamp = timestamp;

  const totalDuration = CHART.total_timesteps * secPerSubdiv;
  if (currentTime > totalDuration + 2) {{
    resetChart();
    return;
  }}

  // Progress
  const progress = Math.min(1, currentTime / totalDuration);
  document.getElementById('progress-fill').style.width = (progress * 100).toFixed(1) + '%';

  // Beat counter
  const beat = Math.floor(currentTime / (60/BPM));
  document.getElementById('beat-counter').textContent = 'Beat: ' + beat;
  beatPulse(currentTime / (60/BPM));

  // Update existing arrow positions
  for (let i = activeArrows.length - 1; i >= 0; i--) {{
    const a = activeArrows[i];
    const y = eventToY(a.timeSec);
    a.el.style.top = (y - 27) + 'px';

    // Flash receptor when arrow crosses hit zone
    if (!a.fired && y >= HIT_Y - 4) {{
      a.fired = true;
      flashReceptor(a.col);
      combo++;
      hitsShown++;
      updateStats();
    }}

    // Remove if off screen
    if (y > STAGE_H + 60) {{
      canvas.removeChild(a.el);
      activeArrows.splice(i, 1);
    }}
  }}

  // Spawn new arrows that should be visible now
  for (let ei = lastRenderedEvent + 1; ei < events.length; ei++) {{
    const ev = events[ei];
    const y = eventToY(ev.timeSec);
    if (y > STAGE_H + 60) break;  // too far in future
    if (y < -60) {{ lastRenderedEvent = ei; continue; }}  // too far past

    // Spawn arrows for each active column
    for (let col = 0; col < 4; col++) {{
      if (!ev.arrows[col]) continue;
      const el = document.createElement('div');
      el.className = 'arrow';
      el.textContent = COL_ARROWS[col];
      el.style.left = COL_XS[col] + 'px';
      el.style.top = (y - 27) + 'px';
      el.style.background = COL_COLORS[col] + '33';
      el.style.borderColor = COL_COLORS[col];
      el.style.color = COL_COLORS[col];
      canvas.appendChild(el);
      activeArrows.push({{ el, timeSec: ev.timeSec, col, fired: false }});
    }}
    lastRenderedEvent = ei;
  }}

  requestAnimationFrame(frame);
}}

// Init
resetChart();
</script>
</body>
</html>"""
    return html


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
