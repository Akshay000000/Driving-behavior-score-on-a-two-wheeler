import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine
} from "recharts";

/* ─────────────────────────────────────────────
   DESIGN SYSTEM — Military HUD / ADAS Terminal
   Amber phosphor on near-black carbon.
   Monospace data, angular cuts, scan-lines.
───────────────────────────────────────────── */
const T = {
  bg:       "#080a08",
  surface:  "#0d0f0d",
  panel:    "#111411",
  border:   "#1a2018",
  amber:    "#ffb300",
  amberDim: "#ff8f0022",
  green:    "#00e676",
  red:      "#ff1744",
  orange:   "#ff6d00",
  cyan:     "#00e5ff",
  text:     "#d4e0c8",
  dim:      "#4a5e44",
  scan:     "repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.08) 2px,rgba(0,0,0,0.08) 4px)",
};

const RIDER_PALETTES = [
  { primary: "#ffb300", glow: "#ffb30033" },
  { primary: "#00e676", glow: "#00e67633" },
  { primary: "#00e5ff", glow: "#00e5ff33" },
  { primary: "#ff6d00", glow: "#ff6d0033" },
  { primary: "#d500f9", glow: "#d500f933" },
  { primary: "#76ff03", glow: "#76ff0333" },
];

const scoreCol = s => s >= 80 ? T.green : s >= 50 ? T.amber : T.red;

/* ── Hex clip-path corners ─────────────────── */
const clip = "polygon(8px 0%, 100% 0%, 100% calc(100% - 8px), calc(100% - 8px) 100%, 0% 100%, 0% 8px)";
const clipSm = "polygon(5px 0%, 100% 0%, 100% calc(100% - 5px), calc(100% - 5px) 100%, 0% 100%, 0% 5px)";

/* ── Animated score arc ────────────────────── */
function ScoreArc({ score, size = 88, color }) {
  const r = size * 0.38;
  const circ = 2 * Math.PI * r;
  const arc = (score / 100) * circ * 0.75; // 270° sweep
  const offset = circ * 0.125;             // start at 7 o'clock
  return (
    <svg width={size} height={size}>
      {/* Track */}
      <circle cx={size/2} cy={size/2} r={r}
        fill="none" stroke={T.border} strokeWidth={5}
        strokeDasharray={`${circ*0.75} ${circ*0.25}`}
        strokeDashoffset={-offset} strokeLinecap="round" />
      {/* Value */}
      <circle cx={size/2} cy={size/2} r={r}
        fill="none" stroke={color} strokeWidth={5}
        strokeDasharray={`${arc} ${circ - arc}`}
        strokeDashoffset={-offset} strokeLinecap="round"
        filter={`drop-shadow(0 0 4px ${color})`}
        style={{ transition: "stroke-dasharray 0.5s cubic-bezier(.4,0,.2,1)" }} />
      {/* Value text */}
      <text x={size/2} y={size/2-4} textAnchor="middle"
        fill={color} fontSize={size*0.24} fontWeight={700}
        fontFamily="'Share Tech Mono',monospace">
        {Math.round(score)}
      </text>
      <text x={size/2} y={size/2+10} textAnchor="middle"
        fill={T.dim} fontSize={size*0.11}
        fontFamily="'Share Tech Mono',monospace">
        /100
      </text>
    </svg>
  );
}

/* ── Animated bar ──────────────────────────── */
function Bar({ label, value, max, color, unit="" }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display:"flex", justifyContent:"space-between",
                    fontSize: 10, color: T.dim, marginBottom: 3,
                    fontFamily:"'Share Tech Mono',monospace", letterSpacing:1 }}>
        <span>{label}</span>
        <span style={{ color: value > 0 ? color : T.dim }}>
          {value}{unit}
        </span>
      </div>
      <div style={{ height: 3, background: T.border, clipPath: clipSm }}>
        <div style={{
          height:"100%", width:`${pct}%`, background: color,
          boxShadow: `0 0 6px ${color}`,
          transition: "width 0.4s cubic-bezier(.4,0,.2,1)"
        }} />
      </div>
    </div>
  );
}

/* ── Rider card ────────────────────────────── */
function RiderCard({ rider, palette, history }) {
  const { primary: col, glow } = palette;
  const helm = rider.helmet === "HELMET"    ? { sym:"EQUIPPED", c: T.green }
             : rider.helmet === "NO_HELMET" ? { sym:"ABSENT",   c: T.red }
             :                                { sym:"SCANNING",  c: T.dim };

  return (
    <div style={{
      background: T.panel,
      clipPath: clip,
      border: `1px solid ${col}33`,
      marginBottom: 10,
      boxShadow: `inset 0 0 30px ${glow}, 0 0 1px ${col}44`,
      position: "relative",
      overflow: "hidden",
    }}>
      {/* Scan line texture */}
      <div style={{ position:"absolute", inset:0, background: T.scan,
                    pointerEvents:"none", opacity:0.4 }} />

      {/* Corner accent */}
      <div style={{ position:"absolute", top:0, right:0, width:16, height:16,
                    background: col, clipPath:"polygon(100% 0,0 0,100% 100%)",
                    opacity:0.7 }} />

      <div style={{ padding: "12px 14px", position:"relative" }}>
        {/* Top row */}
        <div style={{ display:"flex", alignItems:"center", gap:12,
                      marginBottom:10 }}>
          <ScoreArc score={rider.score} size={80} color={col} />
          <div style={{ flex:1 }}>
            <div style={{ display:"flex", justifyContent:"space-between",
                          alignItems:"flex-start" }}>
              <span style={{ fontFamily:"'Share Tech Mono',monospace",
                             fontSize:11, color: col, letterSpacing:2 }}>
                UNIT_{String(rider.id).padStart(2,"0")}
              </span>
              <span style={{
                fontFamily:"'Share Tech Mono',monospace", fontSize:9,
                padding:"2px 6px",
                background: scoreCol(rider.score) + "22",
                color: scoreCol(rider.score),
                clipPath: clipSm,
                letterSpacing:2,
              }}>
                {rider.rating}
              </span>
            </div>

            {/* Speed */}
            <div style={{ marginTop:4 }}>
              <span style={{ fontSize:28, fontWeight:700, color: T.text,
                             fontFamily:"'Share Tech Mono',monospace",
                             lineHeight:1 }}>
                {rider.speed_kmh.toFixed(1)}
              </span>
              <span style={{ fontSize:10, color: T.dim, marginLeft:4 }}>km/h</span>
            </div>

            {/* Tags */}
            <div style={{ display:"flex", gap:6, marginTop:6, flexWrap:"wrap" }}>
              <Tag label={rider.direction} color={col} />
              {rider.stationary && <Tag label="HALT" color={T.orange} />}
              {rider.stunt      && <Tag label="⚡STUNT" color={T.red} blink />}
              <Tag label={`HLMT: ${helm.sym}`} color={helm.c} />
            </div>
          </div>
        </div>

        {/* Event bars */}
        <div style={{ marginBottom:8 }}>
          <Bar label="HARD BRAKE"   value={rider.hard_brake}  max={10} color={T.red}    />
          <Bar label="AGGR ACCEL"   value={rider.aggr_accel}  max={10} color={T.orange} />
          <Bar label="LANE WEAVE"   value={rider.weave}       max={10} color={T.amber}  />
          <Bar label="TAILGATE"     value={rider.tailgate}    max={10} color={T.red}    />
          <Bar label="SUDDEN STOP"  value={rider.sudden_stop} max={10} color={T.orange} />
          <Bar label="NO-HELMET"    value={rider.no_helmet}   max={10} color={T.red}    />
        </div>

        {/* Mini chart */}
        {history && history.length > 3 && (
          <div style={{ height:44, marginBottom:4 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <Line type="monotone" dataKey="speed" stroke={col}
                      dot={false} strokeWidth={1.5} />
                <Line type="monotone" dataKey="score" stroke={T.dim}
                      dot={false} strokeWidth={1} strokeDasharray="2 2" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Reliability */}
        {rider.reliability < 100 && (
          <div style={{ marginTop:4 }}>
            <div style={{ fontSize:9, color:T.dim, marginBottom:2,
                          fontFamily:"'Share Tech Mono',monospace", letterSpacing:1 }}>
              LOCK {Math.round(rider.reliability)}%
            </div>
            <div style={{ height:2, background:T.border }}>
              <div style={{ height:"100%", width:`${rider.reliability}%`,
                            background: T.amber, transition:"width 0.3s" }} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Tag({ label, color, blink }) {
  return (
    <span style={{
      fontSize:8, fontFamily:"'Share Tech Mono',monospace",
      letterSpacing:1.5, padding:"2px 5px",
      background: color + "18", color,
      clipPath: clipSm,
      animation: blink ? "blink 0.8s step-end infinite" : "none",
    }}>
      {label}
    </span>
  );
}

/* ── Fleet status bar ──────────────────────── */
function FleetBar({ riders }) {
  const avg = riders.length
    ? (riders.reduce((s,r) => s+r.score, 0) / riders.length).toFixed(1)
    : "--";
  const avgSpd = riders.length
    ? (riders.reduce((s,r) => s+r.speed_kmh, 0) / riders.length).toFixed(1)
    : "--";
  const noHelmet = riders.filter(r => r.helmet === "NO_HELMET").length;

  return (
    <div style={{
      display:"flex", gap:24, padding:"8px 20px",
      background: T.panel, borderBottom:`1px solid ${T.border}`,
      fontFamily:"'Share Tech Mono',monospace",
    }}>
      <Metric label="ACTIVE" value={riders.length} color={T.cyan} />
      <Metric label="AVG SCORE" value={avg} color={scoreCol(parseFloat(avg)||0)} />
      <Metric label="AVG SPEED" value={`${avgSpd} km/h`} color={T.amber} />
      <Metric label="NO HELMET" value={noHelmet}
              color={noHelmet > 0 ? T.red : T.dim} />
    </div>
  );
}

function Metric({ label, value, color }) {
  return (
    <div style={{ display:"flex", alignItems:"center", gap:8 }}>
      <span style={{ fontSize:9, color:T.dim, letterSpacing:2 }}>{label}</span>
      <span style={{ fontSize:14, color, fontWeight:700 }}>{value}</span>
    </div>
  );
}

/* ── Session report ────────────────────────── */
function SessionReport({ data }) {
  if (!data?.riders?.length) return (
    <div style={{ color:T.dim, textAlign:"center", padding:60,
                  fontFamily:"'Share Tech Mono',monospace", letterSpacing:2 }}>
      STOP SESSION TO GENERATE REPORT
    </div>
  );

  const sorted = [...data.riders].sort((a,b) => b.score - a.score);

  const TH = ({c}) => (
    <th style={{ padding:"8px 10px", textAlign:"left", fontSize:9,
                 color: T.dim, letterSpacing:2,
                 fontFamily:"'Share Tech Mono',monospace",
                 borderBottom:`1px solid ${T.border}` }}>{c}</th>
  );
  const TD = ({c, col}) => (
    <td style={{ padding:"8px 10px", fontSize:12, color: col||T.text,
                 fontFamily:"'Share Tech Mono',monospace",
                 borderBottom:`1px solid ${T.border}18` }}>{c}</td>
  );

  return (
    <div style={{ display:"flex", flexDirection:"column", gap:16 }}>
      {/* Summary cards */}
      <div style={{ display:"grid", gridTemplateColumns:"repeat(4,1fr)", gap:12 }}>
        {[
          { l:"VEHICLES TRACKED", v: data.vehicles, c: T.cyan },
          { l:"DURATION",         v: `${data.duration}s`, c: T.amber },
          { l:"AVERAGE SCORE",    v: data.avg_score, c: scoreCol(data.avg_score) },
          { l:"NO-HELMET RIDERS",
            v: data.riders.filter(r=>r.no_helmet>0).length,
            c: data.riders.filter(r=>r.no_helmet>0).length > 0 ? T.red : T.green },
        ].map(m => (
          <div key={m.l} style={{ background:T.panel, clipPath:clip,
                                   border:`1px solid ${T.border}`,
                                   padding:"16px 18px" }}>
            <div style={{ fontSize:9, color:T.dim, letterSpacing:2,
                          fontFamily:"'Share Tech Mono',monospace",
                          marginBottom:6 }}>{m.l}</div>
            <div style={{ fontSize:28, fontWeight:700, color:m.c,
                          fontFamily:"'Share Tech Mono',monospace" }}>{m.v}</div>
          </div>
        ))}
      </div>

      {/* Table */}
      <div style={{ background:T.panel, clipPath:clip,
                    border:`1px solid ${T.border}`, overflow:"hidden" }}>
        <div style={{ padding:"12px 16px", borderBottom:`1px solid ${T.border}`,
                      fontSize:10, color:T.amber, letterSpacing:3,
                      fontFamily:"'Share Tech Mono',monospace" }}>
          FULL SESSION LOG — {data.vehicles} UNITS
        </div>
        <div style={{ overflowX:"auto" }}>
          <table style={{ width:"100%", borderCollapse:"collapse" }}>
            <thead>
              <tr>
                {["ID","SCORE","km/h","BRAKE","ACCEL","WEAVE",
                  "TAILG","STOP","NO-HLM","HELMET","DIR","RATING"]
                  .map(h => <TH key={h} c={h} />)}
              </tr>
            </thead>
            <tbody>
              {sorted.map(r => {
                const p = RIDER_PALETTES[r.id % RIDER_PALETTES.length];
                return (
                  <tr key={r.id}
                      onMouseEnter={e=>e.currentTarget.style.background=T.border+"44"}
                      onMouseLeave={e=>e.currentTarget.style.background="transparent"}
                      style={{ transition:"background 0.15s", cursor:"default" }}>
                    <TD c={`#${String(r.id).padStart(2,"0")}`} col={p.primary} />
                    <TD c={r.score}   col={scoreCol(r.score)} />
                    <TD c={r.speed_kmh} />
                    <TD c={r.hard_brake}  col={r.hard_brake>0?T.red:T.dim} />
                    <TD c={r.aggr_accel}  col={r.aggr_accel>0?T.orange:T.dim} />
                    <TD c={r.weave}       col={r.weave>0?T.amber:T.dim} />
                    <TD c={r.tailgate}    col={r.tailgate>0?T.red:T.dim} />
                    <TD c={r.sudden_stop} col={r.sudden_stop>0?T.red:T.dim} />
                    <TD c={r.no_helmet}   col={r.no_helmet>0?T.red:T.dim} />
                    <TD c={r.helmet==="HELMET"?"✓":r.helmet==="NO_HELMET"?"✗":"?"}
                        col={r.helmet==="HELMET"?T.green:r.helmet==="NO_HELMET"?T.red:T.dim} />
                    <TD c={r.direction} />
                    <TD c={r.rating} col={scoreCol(r.score)} />
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

/* ── Main dashboard ────────────────────────── */
export default function Dashboard() {
  const [connected,  setConnected]  = useState(false);
  const [running,    setRunning]    = useState(false);
  const [mode,       setMode]       = useState("dashcam");
  const [videoPath,  setVideoPath]  = useState(String.raw`../public/video.mp4`);
  const [riders,     setRiders]     = useState([]);
  const [session,    setSession]    = useState(null);
  const [tab,        setTab]        = useState("live");
  const [riderHist,  setRiderHist]  = useState({});
  const [frameN,     setFrameN]     = useState(0);

  const canvasRef   = useRef(null);
  const wsRef       = useRef(null);
  const pendingRef  = useRef(null);  // pending base64 frame
  const rafRef      = useRef(null);  // requestAnimationFrame id

  /* ── 60fps canvas renderer ─────────────────────────────────────────────── */
  const imgRef = useRef(new window.Image());

  const renderLoop = useCallback(() => {
    const b64 = pendingRef.current;
    if (b64 && canvasRef.current) {
      const img = imgRef.current;
      if (img.src !== "data:image/jpeg;base64," + b64) {
        img.onload = () => {
          const ctx = canvasRef.current?.getContext("2d");
          if (ctx) ctx.drawImage(img, 0, 0,
            canvasRef.current.width, canvasRef.current.height);
        };
        img.src = "data:image/jpeg;base64," + b64;
      }
    }
    rafRef.current = requestAnimationFrame(renderLoop);
  }, []);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(renderLoop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [renderLoop]);

  /* ── WebSocket ─────────────────────────────────────────────────────────── */
  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");
    wsRef.current = ws;
    ws.onopen  = () => setConnected(true);
    ws.onclose = () => { setConnected(false); setRunning(false); };
    ws.onmessage = (evt) => {
      const data = JSON.parse(evt.data);
      if (data.frame) pendingRef.current = data.frame;  // RAF picks it up
      if (data.riders) {
        setRiders(data.riders);
        setFrameN(data.n || 0);
        setRiderHist(prev => {
          const next = { ...prev };
          data.riders.forEach(r => {
            if (!next[r.id]) next[r.id] = [];
            next[r.id] = [...next[r.id].slice(-90),
                          { speed: r.speed_kmh, score: r.score }];
          });
          return next;
        });
      }
    };
    return () => ws.close();
  }, []);

  const handleStart = async () => {
    const res = await fetch("http://localhost:8000/api/start", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ mode, video_path: videoPath }),
    });
    if (res.ok) setRunning(true);
  };

  const handleStop = async () => {
    await fetch("http://localhost:8000/api/stop", { method:"POST" });
    setRunning(false);
    const res  = await fetch("http://localhost:8000/api/session");
    const data = await res.json();
    setSession(data);
    setTab("report");
  };

  /* ── Layout ────────────────────────────────────────────────────────────── */
  return (
    <div style={{
      minHeight:"100vh", background: T.bg, color: T.text,
      fontFamily:"'DM Mono','Share Tech Mono',monospace",
      display:"flex", flexDirection:"column",
      backgroundImage: T.scan,
    }}>

      {/* ── Header ── */}
      <header style={{
        display:"flex", alignItems:"center", justifyContent:"space-between",
        padding:"0 24px", height:52,
        background: T.panel,
        borderBottom:`1px solid ${T.amber}33`,
        position:"sticky", top:0, zIndex:100,
        boxShadow:`0 1px 0 ${T.amber}22, 0 4px 20px #000a`,
      }}>
        {/* Logo */}
        <div style={{ display:"flex", alignItems:"center", gap:12 }}>
          <div style={{
            width:32, height:32, clipPath:clip,
            background: T.amber + "22",
            border:`1px solid ${T.amber}66`,
            display:"flex", alignItems:"center", justifyContent:"center",
            fontSize:16,
          }}>▸</div>
          <div>
            <div style={{ fontSize:13, fontWeight:700, color: T.amber,
                          letterSpacing:3 }}>VARROC EUREKA 3.0</div>
            <div style={{ fontSize:9, color: T.dim, letterSpacing:4 }}>
              PS3 · DRIVING BEHAVIOR SYSTEM
            </div>
          </div>
        </div>

        {/* Controls */}
        <div style={{ display:"flex", alignItems:"center", gap:10 }}>
          {/* Status */}
          <div style={{ display:"flex", alignItems:"center", gap:5,
                        fontSize:9, color: connected ? T.green : T.red,
                        letterSpacing:2 }}>
            <div style={{
              width:6, height:6, borderRadius:"50%",
              background: connected ? T.green : T.red,
              boxShadow: connected ? `0 0 8px ${T.green}` : "none",
            }} />
            {connected ? "LINK" : "OFFLINE"}
          </div>

          <div style={{ width:1, height:20, background: T.border }} />

          {/* Mode toggle */}
          <div style={{ display:"flex", background: T.bg, clipPath:clipSm,
                        border:`1px solid ${T.border}`, overflow:"hidden" }}>
            {["dashcam","fixed"].map(m => (
              <button key={m} onClick={() => setMode(m)} style={{
                padding:"5px 12px", border:"none", cursor:"pointer",
                background: mode===m ? T.amber+"33" : "transparent",
                color: mode===m ? T.amber : T.dim,
                fontSize:9, letterSpacing:2,
                transition:"all 0.2s",
              }}>
                {m.toUpperCase()}
              </button>
            ))}
          </div>

          {/* Path input */}
          <input value={videoPath} onChange={e=>setVideoPath(e.target.value)}
            style={{
              background: T.bg, color: T.dim,
              border:`1px solid ${T.border}`, clipPath:clipSm,
              padding:"5px 10px", fontSize:10, width:260,
              outline:"none", letterSpacing:0.5,
            }} />

          {/* Start / Stop */}
          <HudButton label="▶ START" color={T.green}
            onClick={handleStart} disabled={running || !connected} />
          <HudButton label="■ STOP" color={T.red}
            onClick={handleStop} disabled={!running} />
        </div>
      </header>

      {/* ── Fleet bar ── */}
      {riders.length > 0 && <FleetBar riders={riders} />}

      {/* ── Tab bar ── */}
      <div style={{ display:"flex", gap:2, padding:"10px 24px 0",
                    borderBottom:`1px solid ${T.border}` }}>
        {[["live","▸ LIVE FEED"],["report","▪ SESSION REPORT"]].map(([id,lbl])=>(
          <button key={id} onClick={()=>setTab(id)} style={{
            padding:"7px 18px", border:"none", cursor:"pointer",
            background: tab===id ? T.amber+"22" : "transparent",
            color: tab===id ? T.amber : T.dim,
            fontSize:10, letterSpacing:2,
            clipPath: tab===id ? clip : "none",
            borderBottom: tab===id ? `2px solid ${T.amber}` : "2px solid transparent",
            transition:"all 0.2s",
          }}>{lbl}</button>
        ))}
        {running && (
          <div style={{ marginLeft:"auto", display:"flex", alignItems:"center",
                        gap:6, fontSize:9, color: T.red, letterSpacing:3,
                        animation:"blink 1s step-end infinite" }}>
            <div style={{ width:6,height:6,borderRadius:"50%",
                          background:T.red, boxShadow:`0 0 8px ${T.red}` }} />
            REC · FRAME {frameN}
          </div>
        )}
      </div>

      {/* ── Live feed ── */}
      {tab === "live" && (
        <div style={{ display:"grid", gridTemplateColumns:"1fr 300px",
                      gap:16, padding:"16px 24px", flex:1 }}>

          {/* Left column */}
          <div style={{ display:"flex", flexDirection:"column", gap:14 }}>
            {/* Video */}
            <div style={{
              position:"relative", clipPath:clip,
              border:`1px solid ${T.amber}33`,
              background:"#000",
              boxShadow:`0 0 30px #000, inset 0 0 40px ${T.amber}08`,
            }}>
              <canvas ref={canvasRef} width={640} height={360}
                style={{ width:"100%", display:"block" }} />

              {/* Corner decorations */}
              {[[0,0,"top:0;left:0"],[0,1,"top:0;right:0"],
                [1,0,"bottom:0;left:0"],[1,1,"bottom:0;right:0"]].map(([_,__,pos],i)=>(
                <div key={i} style={{
                  position:"absolute", width:16, height:16,
                  ...Object.fromEntries(pos.split(";").map(p=>{
                    const [k,v]=p.split(":"); return [k,v];
                  })),
                  borderTop: i < 2 ? `2px solid ${T.amber}` : "none",
                  borderBottom: i >= 2 ? `2px solid ${T.amber}` : "none",
                  borderLeft: i%2===0 ? `2px solid ${T.amber}` : "none",
                  borderRight: i%2===1 ? `2px solid ${T.amber}` : "none",
                  opacity:0.7,
                }} />
              ))}

              {!running && (
                <div style={{
                  position:"absolute", inset:0,
                  display:"flex", flexDirection:"column",
                  alignItems:"center", justifyContent:"center",
                  background:"#000c",
                }}>
                  <div style={{ fontSize:32, color:T.amber,
                                opacity:0.3, marginBottom:8 }}>▸</div>
                  <div style={{ fontSize:10, color:T.dim, letterSpacing:4 }}>
                    AWAITING FEED
                  </div>
                </div>
              )}

              {/* Scan overlay */}
              <div style={{ position:"absolute", inset:0, background: T.scan,
                            pointerEvents:"none", opacity:0.3 }} />
            </div>

            {/* Charts */}
            {Object.keys(riderHist).length > 0 && (
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14 }}>
                <ChartPanel title="SPEED HISTORY · km/h" data={riderHist}
                            dataKey="speed" yDomain={[0,80]} />
                <ChartPanel title="SCORE HISTORY" data={riderHist}
                            dataKey="score" yDomain={[0,100]}
                            reference={80} />
              </div>
            )}
          </div>

          {/* Right — rider cards */}
          <div style={{ overflowY:"auto", maxHeight:"calc(100vh - 140px)",
                        paddingRight:4 }}>
            {riders.length === 0 ? (
              <div style={{ color:T.dim, textAlign:"center", padding:40,
                            fontSize:10, letterSpacing:3 }}>
                NO UNITS DETECTED
              </div>
            ) : (
              [...riders].sort((a,b)=>b.score-a.score).map(r => (
                <RiderCard key={r.id} rider={r}
                           palette={RIDER_PALETTES[r.id % RIDER_PALETTES.length]}
                           history={riderHist[r.id]} />
              ))
            )}
          </div>
        </div>
      )}

      {/* ── Report ── */}
      {tab === "report" && (
        <div style={{ padding:"16px 24px" }}>
          <SessionReport data={session} />
        </div>
      )}

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=DM+Mono:wght@300;400;500&display=swap');
        * { box-sizing:border-box; margin:0; padding:0; }
        body { background:${T.bg}; }
        ::-webkit-scrollbar { width:4px; }
        ::-webkit-scrollbar-track { background:${T.bg}; }
        ::-webkit-scrollbar-thumb { background:${T.amber}44; border-radius:2px; }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
        @keyframes fadeIn { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:none} }
        input:focus { outline:none; border-color:${T.amber}66 !important; }
        button:hover:not(:disabled) { filter:brightness(1.15); }
        button:disabled { opacity:0.35 !important; cursor:not-allowed !important; }
      `}</style>
    </div>
  );
}

/* ── HUD button ────────────────────────────── */
function HudButton({ label, color, onClick, disabled }) {
  return (
    <button onClick={onClick} disabled={disabled} style={{
      padding:"6px 16px", border:`1px solid ${color}55`,
      clipPath: clipSm, cursor:"pointer",
      background: color + "18", color,
      fontSize:10, letterSpacing:2,
      transition:"all 0.2s",
    }}>
      {label}
    </button>
  );
}

/* ── Chart panel ───────────────────────────── */
function ChartPanel({ title, data, dataKey, yDomain, reference }) {
  return (
    <div style={{
      background: T.panel, clipPath:clip,
      border:`1px solid ${T.border}`, padding:"12px 14px",
    }}>
      <div style={{ fontSize:9, color:T.dim, letterSpacing:2, marginBottom:8 }}>
        {title}
      </div>
      <ResponsiveContainer width="100%" height={110}>
        <LineChart>
          <XAxis dataKey="n" hide />
          <YAxis domain={yDomain}
            tick={{ fill:T.dim, fontSize:9, fontFamily:"'Share Tech Mono',monospace" }}
            width={28} />
          <Tooltip
            contentStyle={{ background:T.panel, border:`1px solid ${T.border}`,
                            fontSize:10, fontFamily:"'Share Tech Mono',monospace" }}
            labelStyle={{ display:"none" }} />
          {reference && (
            <ReferenceLine y={reference} stroke={T.dim} strokeDasharray="4 4" />
          )}
          {Object.entries(data).map(([id, hist]) => (
            <Line key={id} data={hist} type="monotone" dataKey={dataKey}
                  name={`#${id}`}
                  stroke={RIDER_PALETTES[parseInt(id) % RIDER_PALETTES.length].primary}
                  dot={false} strokeWidth={1.5}
                  filter={`drop-shadow(0 0 3px ${RIDER_PALETTES[parseInt(id) % RIDER_PALETTES.length].primary}88)`}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}