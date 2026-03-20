"""
WSN Animation Generator — ECE 659 Pothole Detection Project

Reads wsn_animation_data.json + vehicle GPS CSVs and generates a
SINGLE self-contained HTML file that can be opened in any browser.

The animation:
  - Runs in real simulation time (5 min of data = 5 min playback at 1x)
  - Speed slider: 1x .. 120x
  - Vehicles move as colored dots on a real OpenStreetMap
  - Green flash lines = successful packet transmissions
  - Red flash lines   = failed packets (channel loss / Markov bad state)
  - CH markers show live queue length
  - Pothole warning icons appear when Base Station detects them
  - Live statistics panel: PDR, packets sent, potholes found

Run from anywhere:
    python wsn_animate.py
"""

import json
import os
import pandas as pd
import numpy as np

_HERE     = os.path.dirname(os.path.abspath(__file__))
ANIM_DATA = os.path.join(_HERE, "wsn_animation_data.json")
DATA_DIR  = os.path.join(_HERE, "simulated_trips")
OUT_HTML  = os.path.join(_HERE, "wsn_animation.html")

TRIP_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]


# ─── Load vehicle GPS tracks ──────────────────────────────────────────────────

def load_tracks(t_start):
    """Return list of 50 track arrays: [[t_rel, lat, lon], ...], one per vehicle."""
    tracks = []
    for trip_id in range(1, 6):
        for veh_id in range(1, 11):
            path = os.path.join(DATA_DIR,
                                f"trip{trip_id}_vehicle{veh_id}.csv")
            if not os.path.exists(path):
                tracks.append([])
                continue
            df = pd.read_csv(path,
                             usecols=["timestamp", "latitude", "longitude"])
            df = df.sort_values("timestamp").iloc[::4].reset_index(drop=True)
            pts = [
                [round(float(r.timestamp) - t_start, 2),
                 round(float(r.latitude),  6),
                 round(float(r.longitude), 6)]
                for r in df.itertuples(index=False)
            ]
            tracks.append(pts)
    return tracks


# ─── Build JavaScript data blob ───────────────────────────────────────────────

def build_data(anim, tracks):
    t_start = anim["t_start"]

    def rel(t):
        return round(float(t) - t_start, 2)

    # TX events: [t_rel, from_lat, from_lon, to_lat, to_lon, ch_id, success]
    tx = sorted(
        [
            [rel(e["t"]),
             round(e["from_lat"], 6), round(e["from_lon"], 6),
             round(e["to_lat"],   6), round(e["to_lon"],   6),
             e["ch_id"],
             1 if e["success"] else 0]
            for e in anim.get("tx_events", [])
        ],
        key=lambda x: x[0],
    )

    # Pothole events: [t_rel, lat, lon, ch_id]
    ph = sorted(
        [[rel(e["t"]), round(e["lat"], 6), round(e["lon"], 6), e["ch_id"]]
         for e in anim.get("pothole_events", [])],
        key=lambda x: x[0],
    )

    # Queue events: [t_rel, ch_id, queue_len]  (subsample to keep size down)
    qe_raw = sorted(
        [[rel(e["t"]), e["ch_id"], e["queue_len"]]
         for e in anim.get("queue_events", [])],
        key=lambda x: x[0],
    )
    # Keep at most one event per CH per second
    qe, seen = [], {}
    for ev in qe_raw:
        key = (ev[1], int(ev[0]))
        if key not in seen:
            seen[key] = True
            qe.append(ev)

    vehicle_colors = []
    for t in range(5):
        for _ in range(10):
            vehicle_colors.append(TRIP_COLORS[t])

    return {
        "duration":       round(anim["t_end"] - t_start, 1),
        "ch_positions":   anim["ch_positions"],
        "vehicle_tracks": tracks,
        "vehicle_colors": vehicle_colors,
        "vehicle_trips":  [(i // 10) + 1 for i in range(50)],
        "tx_events":      tx,
        "pothole_events": ph,
        "queue_events":   qe,
    }


# ─── HTML template ────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ECE659 WSN Pothole Detection — Live Simulation</title>
<link rel="stylesheet"
      href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#0d1117; font-family:'Segoe UI',sans-serif; }
#map { position:absolute; top:0; left:0; width:100%; height:100%; z-index:1; }

/* ── Control panel ── */
#ctrl {
  position:fixed; top:12px; right:12px; z-index:1000;
  background:rgba(13,17,40,0.93); color:#dce8ff;
  border-radius:12px; padding:14px 18px; min-width:250px;
  border:1px solid rgba(80,120,255,0.35);
  box-shadow:0 4px 24px rgba(0,0,0,0.6);
}
#ctrl h3 { font-size:11px; color:#7eb8f7; letter-spacing:1px; margin-bottom:8px; }
#clock { font-size:28px; font-weight:700; color:#fff; font-variant-numeric:tabular-nums; letter-spacing:2px; }
#prog-wrap {
  width:100%; height:6px; background:rgba(255,255,255,0.1);
  border-radius:3px; margin:8px 0; cursor:pointer; position:relative;
}
#prog-bar { height:100%; width:0%; background:#4e9af1; border-radius:3px; }
.row { display:flex; align-items:center; gap:8px; margin-top:8px; }
#btn-play {
  background:#4e9af1; border:none; color:#fff;
  padding:6px 18px; border-radius:6px;
  cursor:pointer; font-size:13px; font-weight:700;
}
#btn-play:hover { background:#3a8be0; }
#spd-lbl { font-size:12px; color:#aac4ee; white-space:nowrap; min-width:72px; }
#spd { flex:1; accent-color:#4e9af1; }

/* ── Stats panel ── */
#stats {
  position:fixed; bottom:16px; left:12px; z-index:1000;
  background:rgba(13,17,40,0.9); color:#dce8ff;
  border-radius:12px; padding:12px 16px; min-width:210px;
  border:1px solid rgba(80,120,255,0.3); font-size:13px; line-height:2;
}
#stats h3 { font-size:11px; color:#7eb8f7; letter-spacing:1px; margin-bottom:4px; }
.sv { font-weight:700; color:#fff; }
.sp { color:#ff6b6b; }

/* ── Queue panel ── */
#qpanel {
  position:fixed; top:12px; left:12px; z-index:1000;
  background:rgba(13,17,40,0.9); color:#dce8ff;
  border-radius:12px; padding:12px 16px; min-width:180px;
  border:1px solid rgba(80,120,255,0.3); font-size:12px; line-height:2;
}
#qpanel h3 { font-size:11px; color:#7eb8f7; letter-spacing:1px; margin-bottom:4px; }
.qrow { display:flex; justify-content:space-between; gap:12px; }
.qlen { font-weight:700; color:#f28e2b; }

/* ── Legend ── */
#legend {
  position:fixed; bottom:16px; right:12px; z-index:1000;
  background:rgba(13,17,40,0.9); color:#dce8ff;
  border-radius:12px; padding:12px 16px;
  border:1px solid rgba(80,120,255,0.3); font-size:12px; line-height:2;
}
#legend h3 { font-size:11px; color:#7eb8f7; letter-spacing:1px; margin-bottom:4px; }
.dot { display:inline-block; width:10px; height:10px;
       border-radius:50%; margin-right:5px; vertical-align:middle; }
.sq  { display:inline-block; width:10px; height:3px;
       margin-right:5px; vertical-align:middle; }
</style>
</head>
<body>
<div id="map"></div>

<div id="ctrl">
  <h3>ECE659 · WSN POTHOLE DETECTION</h3>
  <div id="clock">00:00</div>
  <div id="prog-wrap"><div id="prog-bar"></div></div>
  <div class="row">
    <button id="btn-play">&#9654; Play</button>
    <span id="spd-lbl">Speed: 30×</span>
    <input id="spd" type="range" min="1" max="120" value="30" step="1"/>
  </div>
</div>

<div id="qpanel">
  <h3>CH QUEUE LENGTH</h3>
  <div id="qrows"></div>
</div>

<div id="stats">
  <h3>LIVE STATISTICS</h3>
  Packets sent: <span id="st-tx"  class="sv">0</span><br>
  Packet loss:  <span id="st-err" class="sv">0</span><br>
  Live PDR:     <span id="st-pdr" class="sv">—</span><br>
  Potholes:     <span id="st-ph"  class="sv sp">0</span>
</div>

<div id="legend">
  <h3>LEGEND</h3>
  <span class="dot" style="background:#4e79a7"></span> Trip 1<br>
  <span class="dot" style="background:#f28e2b"></span> Trip 2<br>
  <span class="dot" style="background:#e15759"></span> Trip 3<br>
  <span class="dot" style="background:#76b7b2"></span> Trip 4<br>
  <span class="dot" style="background:#59a14f"></span> Trip 5<br>
  <span class="dot" style="background:#1f77b4;border-radius:2px"></span> Cluster Head<br>
  <span class="sq"  style="background:#4ef150"></span> TX success<br>
  <span class="sq"  style="background:#ff4444"></span> TX failed<br>
  &#9888; Pothole detected
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
/* ======================================================================
   DATA — injected by wsn_animate.py
   ====================================================================== */
const D = /*DATA_PLACEHOLDER*/null;

/* ======================================================================
   MAP SETUP
   ====================================================================== */
const avgLat = D.ch_positions.reduce((a,p)=>a+p[0],0)/D.ch_positions.length;
const avgLon = D.ch_positions.reduce((a,p)=>a+p[1],0)/D.ch_positions.length;

const map = L.map('map',{zoomControl:false}).setView([avgLat,avgLon],14);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{
  attribution:'© OpenStreetMap contributors', maxZoom:19
}).addTo(map);
L.control.zoom({position:'topleft'}).addTo(map);

/* ======================================================================
   CLUSTER HEAD MARKERS
   ====================================================================== */
const chMarkers = D.ch_positions.map((pos,i)=>{
  const icon = L.divIcon({
    html:`<div style="background:#1f77b4;color:#fff;border-radius:50%;
                width:30px;height:30px;display:flex;align-items:center;
                justify-content:center;font-weight:700;font-size:11px;
                border:2px solid rgba(255,255,255,0.9);
                box-shadow:0 0 8px rgba(30,100,255,0.6)">CH${i}</div>`,
    iconSize:[30,30], iconAnchor:[15,15], className:''
  });
  return L.marker([pos[0],pos[1]],{icon})
          .addTo(map)
          .bindTooltip(`Cluster Head ${i} (Trip ${i+1})`,{permanent:false});
});

/* Build queue panel rows */
const qrowsEl = document.getElementById('qrows');
const qLenEls = D.ch_positions.map((_,i)=>{
  const row = document.createElement('div');
  row.className='qrow';
  row.innerHTML=`<span>CH${i}</span><span class="qlen" id="ql${i}">0</span>`;
  qrowsEl.appendChild(row);
  return document.getElementById(`ql${i}`);
});

/* ======================================================================
   VEHICLE MARKERS
   ====================================================================== */
const vehicleMarkers = D.vehicle_tracks.map((track,i)=>{
  if(!track||track.length===0) return null;
  const c=D.vehicle_colors[i];
  const icon=L.divIcon({
    html:`<div style="width:9px;height:9px;border-radius:50%;background:${c};
               border:1.5px solid rgba(255,255,255,0.8);
               box-shadow:0 0 5px rgba(0,0,0,0.6)"></div>`,
    iconSize:[9,9], iconAnchor:[4,4], className:''
  });
  const first=track[0];
  return L.marker([first[1],first[2]],{icon})
          .addTo(map)
          .bindTooltip(`Trip ${D.vehicle_trips[i]} Veh ${(i%10)+1}`,{permanent:false});
});

/* ======================================================================
   TX LINE POOL
   ====================================================================== */
let txLines=[];
function addTxLine(fl,fn,tl,tn,ok){
  const line=L.polyline([[fl,fn],[tl,tn]],{
    color: ok?'#4ef150':'#ff4444',
    weight:1.5, opacity:0.85, dashArray: ok?null:'4,3'
  }).addTo(map);
  txLines.push({line, exp:simT+1.2});
}
function pruneLines(){
  const keep=[], rm=[];
  txLines.forEach(e=>(e.exp<simT?rm:keep).push(e));
  rm.forEach(e=>map.removeLayer(e.line));
  txLines=keep;
}

/* ======================================================================
   POTHOLE MARKERS
   ====================================================================== */
let phCount=0;
function addPothole(lat,lon){
  const icon=L.divIcon({
    html:`<div style="font-size:24px;filter:drop-shadow(0 0 4px rgba(255,80,0,0.9))">&#9888;</div>`,
    iconSize:[24,24], iconAnchor:[12,12], className:''
  });
  L.marker([lat,lon],{icon}).addTo(map)
   .bindTooltip('Pothole Detected',{permanent:false});
  phCount++;
  document.getElementById('st-ph').textContent=phCount;
}

/* ======================================================================
   VEHICLE INTERPOLATION
   ====================================================================== */
const trkPtrs=new Array(D.vehicle_tracks.length).fill(0);
function getVehPos(i,t){
  const tr=D.vehicle_tracks[i];
  if(!tr||tr.length===0) return null;
  let p=trkPtrs[i];
  while(p<tr.length-1 && tr[p+1][0]<=t) p++;
  trkPtrs[i]=p;
  if(p>=tr.length-1){
    return (t-tr[tr.length-1][0]>30)?null:[tr[tr.length-1][1],tr[tr.length-1][2]];
  }
  const a=tr[p], b=tr[p+1];
  const f=Math.min(1,(t-a[0])/(b[0]-a[0]));
  return [a[1]+f*(b[1]-a[1]), a[2]+f*(b[2]-a[2])];
}

/* ======================================================================
   EVENT POINTERS
   ====================================================================== */
let txPtr=0, phPtr=0, qPtr=0;
let nTx=0, nErr=0;
const chQueueLen=[0,0,0,0,0];

function processEvents(t){
  /* TX events */
  while(txPtr<D.tx_events.length && D.tx_events[txPtr][0]<=t){
    const e=D.tx_events[txPtr++];
    const ok=e[6]===1;
    nTx++; if(!ok) nErr++;
    addTxLine(e[1],e[2],e[3],e[4],ok);
  }
  /* Pothole events */
  while(phPtr<D.pothole_events.length && D.pothole_events[phPtr][0]<=t){
    const e=D.pothole_events[phPtr++];
    addPothole(e[1],e[2]);
  }
  /* Queue events */
  while(qPtr<D.queue_events.length && D.queue_events[qPtr][0]<=t){
    const e=D.queue_events[qPtr++];
    chQueueLen[e[1]]=e[2];
  }
  /* Update stats */
  document.getElementById('st-tx').textContent=nTx;
  document.getElementById('st-err').textContent=nErr;
  document.getElementById('st-pdr').textContent=
    nTx>0?((nTx-nErr)/nTx*100).toFixed(1)+'%':'—';
  chQueueLen.forEach((q,i)=>{ qLenEls[i].textContent=q; });
}

/* ======================================================================
   ANIMATION LOOP
   ====================================================================== */
let simT=0, playing=false, lastWall=null, speedX=30;

function fmt(s){
  const m=Math.floor(s/60), sec=Math.floor(s%60);
  return String(m).padStart(2,'0')+':'+String(sec).padStart(2,'0');
}

function frame(wall){
  if(!playing) return;
  if(lastWall!==null){
    simT+=((wall-lastWall)/1000)*speedX;
    if(simT>=D.duration){
      simT=D.duration; playing=false;
      document.getElementById('btn-play').innerHTML='&#9654; Replay';
    }
  }
  lastWall=wall;

  /* Move vehicles */
  vehicleMarkers.forEach((m,i)=>{
    if(!m) return;
    const pos=getVehPos(i,simT);
    if(pos){ m.setLatLng(pos); if(!map.hasLayer(m)) m.addTo(map); }
    else   { if(map.hasLayer(m)) map.removeLayer(m); }
  });

  processEvents(simT);
  pruneLines();

  document.getElementById('clock').textContent=fmt(simT);
  document.getElementById('prog-bar').style.width=(simT/D.duration*100).toFixed(2)+'%';

  requestAnimationFrame(frame);
}

/* ======================================================================
   CONTROLS
   ====================================================================== */
function resetState(){
  simT=0; txPtr=0; phPtr=0; qPtr=0; nTx=0; nErr=0; phCount=0;
  trkPtrs.fill(0);
  txLines.forEach(e=>map.removeLayer(e.line)); txLines=[];
  chQueueLen.fill(0);
  /* Remove existing pothole markers (DivIcon markers not easily distinguishable,
     so we recreate the full marker layer set on full reset via page reload) */
  document.getElementById('st-ph').textContent='0';
  document.getElementById('st-tx').textContent='0';
  document.getElementById('st-err').textContent='0';
  document.getElementById('st-pdr').textContent='—';
  qLenEls.forEach(el=>{ el.textContent='0'; });
}

document.getElementById('btn-play').addEventListener('click',()=>{
  if(simT>=D.duration){ location.reload(); return; }
  playing=!playing; lastWall=null;
  document.getElementById('btn-play').innerHTML=playing?'&#9646;&#9646; Pause':'&#9654; Play';
  if(playing) requestAnimationFrame(frame);
});

document.getElementById('spd').addEventListener('input',function(){
  speedX=parseInt(this.value);
  document.getElementById('spd-lbl').textContent=`Speed: ${speedX}×`;
});

document.getElementById('prog-wrap').addEventListener('click',function(e){
  const f=(e.clientX-this.getBoundingClientRect().left)/this.offsetWidth;
  simT=f*D.duration; lastWall=null;
  document.getElementById('clock').textContent=fmt(simT);
  document.getElementById('prog-bar').style.width=(f*100).toFixed(2)+'%';
});
</script>
</body>
</html>
"""


def main():
    if not os.path.exists(ANIM_DATA):
        print(f"ERROR: {ANIM_DATA} not found — run wsn_main.py first.")
        return

    print(f"Loading {ANIM_DATA} ...")
    with open(ANIM_DATA) as f:
        anim = json.load(f)

    print("Loading vehicle GPS tracks ...")
    tracks = load_tracks(anim["t_start"])

    print("Building animation data ...")
    data = build_data(anim, tracks)

    # Inject data into HTML template
    data_json = json.dumps(data, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("/*DATA_PLACEHOLDER*/null", data_json)

    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    duration_min = data["duration"] / 60
    n_tx  = len(data["tx_events"])
    n_ph  = len(data["pothole_events"])
    n_suc = sum(1 for e in data["tx_events"] if e[6] == 1)
    pdr   = n_suc / max(n_tx, 1) * 100

    print(f"\nAnimation generated -> {OUT_HTML}")
    print(f"  Simulation duration : {duration_min:.1f} minutes")
    print(f"  TX events           : {n_tx}  ({pdr:.1f}% PDR)")
    print(f"  Pothole events      : {n_ph}")
    print(f"  File size           : {os.path.getsize(OUT_HTML)//1024} KB")
    print(f"\nOpen {OUT_HTML} in Chrome/Firefox to watch the live simulation.")
    print(f"  Default speed: 30x  (real time playback with speed slider)")


if __name__ == "__main__":
    main()
