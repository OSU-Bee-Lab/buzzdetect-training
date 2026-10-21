"""Interactive frontier page: the frontier_svg.py scatter as one self-contained HTML file (no network, no libraries).

Hover a point for its exact model name and numbers; click to pin it and see per-fold sensitivity, loudness tiers and the
jet-fold probe. Colour = trunk width, shape = front-end family, outline = rung, faded = a quarantined (tagged) pool,
dashed outline = a speed timed only before the mmt export (frontier_svg.load_all: speed is each graph's fastest export).
Filters: width, depth, front end, class subset, rung, pool, speed/sensitivity range, name search; the dashed frontier
can be computed over the visible points or over every run. Trunks and front ends are parsed from their names, so a
new one gets a colour and a glyph without an edit here.

    python tools/human/frontier_html.py [out.html]      (train env; frontier_svg.load_all: every rung, clean and tagged pools)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '05_distill'))
import dpaths as D  # noqa: E402
import frontier_svg as F  # noqa: E402
import ladder_record as LR  # noqa: E402

KEEP = ('name', 'frontend', 'arch', 'classes', 'init', 'seed', 'steps', 'headline', 'headline_inclusive', 'x_yamnet200',
        'x_yamnet20', 'export', 'speed_from', 'lost_pct', 'gained_pct', 'mae_live', 'mae_buzz', 'per_fold', 'tiers', 'wall_s', 'rung', 'pool')


def rows():
    out = []
    for r in F.load_all():            # every rung, one point per student and rung (frontier_svg.py)
        d = {k: r.get(k) for k in KEEP}
        d['subset'] = F.subset(r) or 'all'
        d['probe'] = F.probe_of(r['name'])
        d['student'] = '|'.join(map(str, F.ident(r)[:-1]))      # one student across rungs
        d['pre_mmt'] = F.pre_mmt(r)
        out.append(d)
    return out


def refs():
    """Horizontal reference lines: the era baseline and the teacher scored by the students' own harness."""
    out = []
    for (name, p), short in zip(LR.REFS, ('era baseline', 'teacher, honest rotation', 'teacher, same harness')):
        if short != 'teacher, honest rotation' and os.path.exists(p):
            out.append({'label': short, 'title': name, 'v': round(LR.sx(p)[0], 3)})
    return out or [{'label': 'era baseline', 'title': 'era baseline', 'v': F.BASELINE}]


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Distillation frontier</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--grid:#e5e5e5;--box:#999;--panel:#f6f6f6;--hi:#111;--halo:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e8;--mut:#999;--grid:#2c2c2c;--box:#666;--panel:#222;--hi:#fff;--halo:#161616}}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.4 system-ui,sans-serif}
main{max-width:1240px;margin:0 auto;padding:16px}
h1{font-size:16px;margin:0 0 2px}p.sub{margin:0 0 12px;color:var(--mut)}
#layout{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
#controls{flex:0 0 240px;display:flex;flex-direction:column;gap:10px}
fieldset{border:1px solid var(--grid);border-radius:6px;margin:0;padding:6px 10px 8px}
legend{font-weight:600;padding:0 4px}
label{display:flex;align-items:center;gap:6px;cursor:pointer;padding:1px 0}
.fam{color:var(--mut);font-size:11px;margin:4px 0 0}
.cols{display:grid;grid-template-columns:1fr 1fr 1fr;column-gap:4px}
.sw{width:11px;height:11px;border-radius:2px;display:inline-block}
input[type=search]{width:100%;box-sizing:border-box;padding:4px;background:var(--panel);color:var(--fg);border:1px solid var(--grid);border-radius:4px}
input[type=range]{width:100%}
button{background:var(--panel);color:var(--fg);border:1px solid var(--box);border-radius:4px;padding:3px 8px;cursor:pointer}
#chartwrap{flex:1 1 640px;min-width:0;position:relative}
svg{width:100%;height:auto;display:block}svg text{fill:var(--fg)}
#legend{display:flex;flex-wrap:wrap;gap:4px 16px;margin:6px 0 0 62px;color:var(--mut);font-size:12px}
#legend span{display:inline-flex;align-items:center;gap:5px}
#tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--box);border-radius:6px;padding:6px 8px;
 font-size:12px;display:none;max-width:360px;box-shadow:0 2px 8px #0004;z-index:2}
#tip b{word-break:break-all}
#detail{margin-top:12px;background:var(--panel);border-radius:6px;padding:8px 12px;min-height:40px}
#detail table{border-collapse:collapse;margin-top:4px}#detail td,#detail th{padding:1px 10px 1px 0;text-align:left;font-weight:normal}
#detail th{color:var(--mut)}
#listwrap{overflow-x:auto}
table.list{border-collapse:collapse;width:100%;margin-top:12px}
table.list th{cursor:pointer;text-align:left;border-bottom:1px solid var(--box);padding:3px 6px;white-space:nowrap}
table.list td{padding:2px 6px;border-bottom:1px solid var(--grid);white-space:nowrap}
table.list tr:hover td{background:var(--panel)}
table.list tr.pin td{background:var(--panel);font-weight:600}
.num{text-align:right;font-variant-numeric:tabular-nums}
.mut{color:var(--mut)}
</style></head><body><main>
<h1>Speed / sensitivity frontier (every rung, seed 1, each student at its stop-rule budget)</h1>
<p class="sub" id="sub"></p>
<div id="layout">
 <div id="controls">
  <fieldset><legend>Trunk width (colour)</legend><div id="f-w"></div></fieldset>
  <fieldset><legend>Trunk depth (layers of 14)</legend><div id="f-d" class="cols"></div></fieldset>
  <fieldset><legend>Front end (shape)</legend><div id="f-fe"></div></fieldset>
  <fieldset><legend>Classes</legend><div id="f-sub"></div></fieldset>
  <fieldset><legend>Rung (outline)</legend><div id="f-rung" class="cols"></div></fieldset>
  <fieldset><legend>Pool</legend><div id="f-pool"></div></fieldset>
  <fieldset><legend>Search</legend><input type="search" id="q" placeholder="model name contains…"></fieldset>
  <fieldset><legend>Min speed: <span id="vx"></span></legend><input type="range" id="minx" step="0.05"></fieldset>
  <fieldset><legend>Min headline: <span id="vy"></span></legend><input type="range" id="miny" step="0.01"></fieldset>
  <fieldset><legend>Display</legend>
   <label><input type="checkbox" id="fr-vis" checked> frontier of visible points</label>
   <label><input type="checkbox" id="fr-all"> also show the frontier of all runs</label>
   <label><input type="checkbox" id="labels" checked> label frontier points</label>
   <label><input type="checkbox" id="join"> join one student across rungs</label>
   <label><input type="checkbox" id="dim"> hide non-frontier points</label>
  </fieldset>
  <button id="reset">Reset filters</button>
 </div>
 <div id="chartwrap"><svg id="chart" viewBox="0 0 860 560"></svg><div id="legend"></div><div id="tip"></div>
  <div id="detail"><span class="mut">Click a point (or a table row) to pin its details.</span></div></div>
</div>
<div id="listwrap"><table class="list" id="list"></table></div>
</main>
<script>
const DATA=__DATA__, REFS=__REFS__, BASELINE=REFS[0].v;
const WIDTH_COLOUR={'0.25':'#d95f02','0.375':'#7570b3','0.50':'#1b9e77','0.75':'#e7298a'};
const SPARE=['#66a61e','#e6ab02','#a6761d','#1f78b4'];
// front-end families: one glyph each, so a new member (yam64h40, fast32h16lo...) inherits its family's shape
const FAMILIES=[['yamnet',/^yamnet$/,'circle'],['yam64 (64 mel)',/^yam64/,'tridown'],['b64',/^b64/,'star'],
 ['twofast32',/^twofast32/,'cross'],['fast32h32',/^fast32h32/,'tri'],['fast32h16',/^fast32h16/,'diamond'],
 ['fast32',/^fast32/,'square'],['two32 / lo32',/^(two32|lo32)/,'hex']];
const family=fe=>FAMILIES.find(([,re])=>re.test(fe))||['other',/./,'circle'];
const RUNG_SW={A:0,B:1.2,C:2.6,D:2.6,E:2.6};
const SUB_NAME={all:'all 15 classes',sub:'buzz + rain + human',buzz:'buzz only'};
const SUB_TAG={all:' all15',sub:'',buzz:' buzz'};
const W=860,H=560,L=62,R=14,T=14,B=48;
const $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';

// trunk 'a<width>[_d<depth>][_f<k>|_l<i>x<w>...]': width colours, depth filters, the rest is a variant tag
for(const d of DATA){const m=/^a([\d.]+)(?:_d(\d+))?(.*)$/.exec(d.arch||'')||[];
 d.width=m[1]?(+m[1]).toFixed(m[1].length>4?3:2):'?';d.depth=m[2]?+m[2]:14;d.variant=(m[3]||'').replace(/^_/,'');
 d.fam=family(d.frontend)[0]}
const uniq=k=>[...new Set(DATA.map(d=>d[k]))];
const widths=uniq('width').sort((a,b)=>a-b);
widths.filter(w=>!WIDTH_COLOUR[w]).forEach((w,i)=>WIDTH_COLOUR[w]=SPARE[i%SPARE.length]);
const colour=d=>WIDTH_COLOUR[d.width]||'#888';
const FILTERS={w:'width',d:'depth',fe:'frontend',sub:'subset',rung:'rung',pool:'pool'};
const state={q:'',pinned:null};
const resetSets=()=>{for(const k in FILTERS)state[k]=new Set(uniq(FILTERS[k]))};resetSets();
const xs=DATA.map(d=>d.x_yamnet200), ys=DATA.map(d=>d.headline), refv=REFS.map(r=>r.v);
const X0=Math.floor(Math.min(...xs,1)*10)/10-0.1, X1=Math.max(...xs)+0.2;
const Y0=Math.floor(Math.min(...ys,...refv)*20)/20-0.02, Y1=Math.ceil(Math.max(...ys,...refv)*20)/20+0.02;
const px=v=>L+(v-X0)/(X1-X0)*(W-L-R), py=v=>H-B-(v-Y0)/(Y1-Y0)*(H-B-T);

function shape(kind,x,y,col,o={}){
 const s=6.5,g=document.createElementNS(NS,'g');let e;
 const mk=(t,a)=>{const n=document.createElementNS(NS,t);for(const k in a)n.setAttribute(k,a[k]);return n};
 const poly=p=>mk('polygon',{points:p.map(q=>q.map(v=>v.toFixed(1)).join(',')).join(' ')});
 if(kind==='square')e=mk('rect',{x:x-s+0.5,y:y-s+0.5,width:2*s-1,height:2*s-1});
 else if(kind==='diamond')e=poly([[x,y-s-2],[x+s+2,y],[x,y+s+2],[x-s-2,y]]);
 else if(kind==='tri')e=poly([[x,y-s-1.5],[x+s+1.5,y+s],[x-s-1.5,y+s]]);
 else if(kind==='tridown')e=poly([[x,y+s+1.5],[x+s+1.5,y-s],[x-s-1.5,y-s]]);
 else if(kind==='hex')e=poly([[x-s,y],[x-s/2,y-s],[x+s/2,y-s],[x+s,y],[x+s/2,y+s],[x-s/2,y+s]]);
 else if(kind==='star'){const p=[];for(let i=0;i<10;i++){const r=i%2?s*0.5:s+2.5,a=Math.PI*i/5-Math.PI/2;p.push([x+r*Math.cos(a),y+r*Math.sin(a)])}e=poly(p)}
 else if(kind==='cross'){const a=2.5;e=poly([[x-a,y-s],[x+a,y-s],[x+a,y-a],[x+s,y-a],[x+s,y+a],[x+a,y+a],[x+a,y+s],[x-a,y+s],[x-a,y+a],[x-s,y+a],[x-s,y-a],[x-a,y-a]])}
 else e=mk('circle',{cx:x,cy:y,r:s});
 e.setAttribute('fill',col);e.setAttribute('fill-opacity',o.faded?'.3':'.88');
 const sw=RUNG_SW[o.rung]||0;
 if(sw){e.setAttribute('stroke','var(--hi)');e.setAttribute('stroke-width',sw);if(o.faded)e.setAttribute('stroke-opacity','.45')}
 else if(o.faded){e.setAttribute('stroke',col);e.setAttribute('stroke-width',1)}
 if(o.slow){e.setAttribute('stroke','var(--hi)');e.setAttribute('stroke-width',Math.max(sw,1.4));e.setAttribute('stroke-dasharray','2.5 1.8')}
 g.appendChild(e);
 if(o.sub==='buzz')g.appendChild(mk('circle',{cx:x,cy:y,r:2.2,fill:'var(--hi)'}));
 if(o.sub==='all')g.appendChild(mk('circle',{cx:x,cy:y,r:2.6,fill:'none',stroke:'var(--hi)','stroke-width':1.3}));
 return g;
}
const mark=(d,x,y)=>shape(family(d.frontend)[2],x,y,colour(d),{rung:d.rung,sub:d.subset,faded:d.pool!=='clean',slow:d.pre_mmt});
function pareto(rows){
 let best=-1;const out=new Set();
 for(const r of [...rows].sort((a,b)=>b.x_yamnet200-a.x_yamnet200||b.headline-a.headline))if(r.headline>best){out.add(r);best=r.headline}
 return out;
}
const f=(v,d=3)=>v==null||v!==v?'–':(+v).toFixed(d);
const trunk=d=>`a${d.width}${d.depth!==14?' d'+d.depth:''}${d.variant?' '+d.variant:''}`;
const short=d=>`${d.frontend} ${trunk(d)}${SUB_TAG[d.subset]}`;
function visible(){
 return DATA.filter(d=>Object.entries(FILTERS).every(([k,c])=>state[k].has(d[c]))
  &&d.name.toLowerCase().includes(state.q)&&d.x_yamnet200>=+$('minx').value&&d.headline>=+$('miny').value);
}
function tipHtml(d){
 return `<b>${d.name}</b><br>${d.frontend} · ${trunk(d)} · ${SUB_NAME[d.subset]}${d.init?' · init '+d.init:''}`
  +`<br>rung ${d.rung} · ${d.steps/1000}k steps · ${d.pool==='clean'?'clean':'<i>'+d.pool+'</i>'}`
  +`<br>headline <b>${f(d.headline)}</b> (${f(d.headline/BASELINE*100,0)}% of baseline)`
  +`<br>speed <b>${f(d.x_yamnet200,2)}×</b> YAMNet (${d.export}${d.pre_mmt?', timed before mmt':''}) · lost buzz ${f(d.lost_pct,1)}%`;
}
const tip=$('tip');
function showTip(ev,d){
 const wr=$('chartwrap').getBoundingClientRect();tip.innerHTML=tipHtml(d);tip.style.display='block';
 let x=ev.clientX-wr.left+14,y=ev.clientY-wr.top+14;
 if(x+tip.offsetWidth>wr.width)x=ev.clientX-wr.left-tip.offsetWidth-14;
 tip.style.left=x+'px';tip.style.top=y+'px';
}
function pin(d){state.pinned=state.pinned===d?null:d;draw();detail()}
function detail(){
 const d=state.pinned,el=$('detail');
 if(!d){el.innerHTML='<span class="mut">Click a point (or a table row) to pin its details.</span>';return}
 const pf=(d.per_fold||[]).map(v=>f(v,2)).join(' · ');
 const tiers=Object.entries(d.tiers||{}).filter(([,v])=>v!=null).map(([k,v])=>`${k} ${f(v,2)}`).join(' · ');
 const p=d.probe, same=DATA.filter(o=>o.student===d.student&&o!==d);
 el.innerHTML=`<b>${d.name}</b> <span class="mut">${d.frontend} · ${trunk(d)} · ${SUB_NAME[d.subset]} · rung ${d.rung} · ${d.pool} · seed ${d.seed} · ${d.steps} steps · wall ${f(d.wall_s/60,0)} min</span>
 <table><tr><th>headline (excl. quiet, fpr 0.005)</th><td>${f(d.headline)}</td><th>incl. quiet</th><td>${f(d.headline_inclusive)}</td></tr>
 <tr><th>speed (200 s / 20 s)</th><td>${f(d.x_yamnet200,2)}× / ${f(d.x_yamnet20,2)}× <span class="mut">${d.export}</span></td><th>buzz lost / gained</th><td>${f(d.lost_pct,1)}% / ${f(d.gained_pct,1)}%</td></tr>
 ${d.speed_from&&d.speed_from!==d.name?`<tr><th>speed timed as</th><td colspan="3">${d.speed_from}</td></tr>`:''}
 <tr><th>logit error (all / buzz)</th><td>${f(d.mae_live)} / ${f(d.mae_buzz)}</td><th></th><td></td></tr>
 <tr><th>per rotating fold</th><td colspan="3">${pf}</td></tr>
 <tr><th>by loudness tier</th><td colspan="3">${tiers}</td></tr>
 <tr><th>jet fold 1_95 FPR</th><td colspan="3">${p&&p.pooled?`all negatives ${f(p.pooled.fpr*100,2)}% · jet frames ${f(p.pooled.fpr_jet*100,2)}% · worst rotating fold ${f(p.pooled.rotating_fpr_max*100,2)}%`:'<span class="mut">no probe.json</span>'}</td></tr>
 ${same.length?`<tr><th>same student, other rungs</th><td colspan="3">${same.map(o=>`rung ${o.rung}: ${f(o.headline)} @ ${f(o.x_yamnet200,2)}×`).join(' · ')}</td></tr>`:''}</table>`;
}
function draw(){
 const svg=$('chart'),vis=visible(),visSet=new Set(vis);svg.innerHTML='';
 const add=(t,a,txt,parent=svg)=>{const n=document.createElementNS(NS,t);for(const k in a)n.setAttribute(k,a[k]);if(txt!=null)n.textContent=txt;parent.appendChild(n);return n};
 for(let v=Math.ceil(Y0*20)/20;v<=Y1+1e-9;v+=0.05){add('line',{x1:L,x2:W-R,y1:py(v),y2:py(v),stroke:'var(--grid)'});add('text',{x:L-8,y:py(v)+4,'text-anchor':'end'},v.toFixed(2))}
 const step=X1-X0>5?1:0.5;
 for(let v=Math.ceil(X0/step)*step;v<X1;v+=step){add('line',{x1:px(v),x2:px(v),y1:T,y2:H-B,stroke:'var(--grid)'});add('text',{x:px(v),y:H-B+16,'text-anchor':'middle'},v+'×')}
 if(X0<1&&X1>1){add('line',{x1:px(1),x2:px(1),y1:T,y2:H-B,stroke:'var(--mut)','stroke-dasharray':'4 3'});add('text',{x:px(1)+4,y:H-B-6,fill:'var(--mut)'},'YAMNet speed')}
 for(const r of REFS){              // label in the emptiest stretch of the line, below it, scanning right to left
  const t=`${r.label} ${f(r.v)}`,w=6.2*t.length,y=py(r.v);
  add('line',{x1:L,x2:W-R,y1:y,y2:y,stroke:'var(--mut)','stroke-dasharray':'4 3'});
  const crowd=x=>vis.filter(d=>{const a=px(d.x_yamnet200),b=py(d.headline);return a>x-w-14&&a<x+14&&b>y-14&&b<y+34}).length;
  let bx=W-R-4;for(let x=W-R-4;x-w>L;x-=10)if(crowd(x)<crowd(bx))bx=x;
  add('text',{x:bx,y:y+14,'text-anchor':'end',fill:'var(--mut)',stroke:'var(--halo)','stroke-width':3,'paint-order':'stroke'},t)
   .appendChild(document.createElementNS(NS,'title')).textContent=r.title}
 add('rect',{x:L,y:T,width:W-L-R,height:H-B-T,fill:'none',stroke:'var(--box)'});
 add('text',{x:(L+W-R)/2,y:H-10,'text-anchor':'middle'},'speed, × YAMNet (GPU, 200 s audio; higher is faster)');
 add('text',{transform:`translate(16 ${(T+H-B)/2}) rotate(-90)`,'text-anchor':'middle'},'sensitivity (excl. quiet) @ 0.5% FPR');
 const frAll=pareto(DATA),frVis=pareto(vis);
 const line=(pts,dash,col,w=1.5)=>{if(pts.length>1)add('polyline',{fill:'none',stroke:col,'stroke-width':w,'stroke-dasharray':dash,
  points:pts.map(r=>px(r.x_yamnet200).toFixed(1)+','+py(r.headline).toFixed(1)).join(' ')})};
 const byX=s=>[...s].sort((a,b)=>b.x_yamnet200-a.x_yamnet200);
 if($('join').checked){const g={};for(const d of vis)(g[d.student]=g[d.student]||[]).push(d);
  for(const k in g)line(g[k].sort((a,b)=>a.rung<b.rung?-1:1),'','var(--box)',1)}
 if($('fr-all').checked)line(byX(frAll),'1 4','var(--mut)');
 if($('fr-vis').checked)line(byX(frVis),'2 3','var(--hi)');
 const dim=$('dim').checked,order=[...vis].sort((a,b)=>(frVis.has(a)-frVis.has(b))||(a.rung<b.rung?-1:1));
 for(const d of order){
  const isFr=frVis.has(d);if(dim&&!isFr)continue;
  const g=mark(d,px(d.x_yamnet200),py(d.headline));
  g.style.cursor='pointer';if(!isFr&&$('fr-vis').checked)g.style.opacity=.8;
  g.addEventListener('mousemove',e=>showTip(e,d));g.addEventListener('mouseleave',()=>tip.style.display='none');
  g.addEventListener('click',()=>pin(d));svg.appendChild(g);
 }
 if($('labels').checked){               // frontier labels: right of the point (left near the edge), pushed apart vertically
  const placed=[];
  for(const d of [...frVis].sort((a,b)=>b.headline-a.headline)){
   const x=px(d.x_yamnet200),y=py(d.headline),t=short(d),w=5.6*t.length,left=x+12+w>W-R-4;
   const x0=left?x-12-w:x+12;let ly=y+4;
   const hit=yy=>placed.some(([a,b,c])=>Math.abs(yy-c)<12&&x0<b&&x0+w>a);
   for(let k=1;hit(ly)&&k<40;k++)ly=y+4+(k%2?1:-1)*Math.ceil(k/2)*12;
   placed.push([x0,x0+w,ly]);
   if(Math.abs(ly-y-4)>7)add('line',{x1:left?x0+w+1:x0-1,y1:ly-4,x2:x+(left?-7:7),y2:y,stroke:'var(--mut)','stroke-width':.6});
   for(const halo of [true,false])add('text',{x:x0,y:ly,'font-size':10.5,...(halo?{stroke:'var(--halo)','stroke-width':3,'stroke-linejoin':'round'}:{})},t);
  }
 }
 if(state.pinned&&visSet.has(state.pinned)){const d=state.pinned;
  add('circle',{cx:px(d.x_yamnet200),cy:py(d.headline),r:13,fill:'none',stroke:'var(--hi)','stroke-width':1.5,'stroke-dasharray':'3 2'})}
 $('sub').textContent=`teacher ${__TEACHER__} · ${vis.length} of ${DATA.length} runs shown · ${frVis.size} on the frontier · hover for the model name, click to pin`;
 table(vis,frVis);
}
let sortKey='headline',sortDir=-1;
const TEXT_COLS=new Set(['name','subset','trunk','frontend','rung','pool']);
function table(vis,fr){
 const cols=[['name','model'],['subset','classes'],['trunk','trunk'],['frontend','front end'],['rung','rung'],['steps','steps'],
  ['pool','pool'],['x_yamnet200','speed ×'],['headline','headline'],['lost_pct','lost %']];
 const val=(d,k)=>k==='trunk'?trunk(d):d[k];
 const rows=[...vis].sort((a,b)=>{const x=val(a,sortKey),y=val(b,sortKey);return (x<y?-1:x>y?1:0)*sortDir});
 $('list').innerHTML='<thead><tr>'+cols.map(([k,t])=>`<th data-k="${k}">${t}${k===sortKey?(sortDir<0?' ▼':' ▲'):''}</th>`).join('')+'<th></th></tr></thead><tbody>'
  +rows.map(d=>`<tr data-i="${DATA.indexOf(d)}"${d===state.pinned?' class="pin"':''}><td>${d.name}</td><td>${SUB_NAME[d.subset]}</td><td>${trunk(d)}</td><td>${d.frontend}</td>`
  +`<td>${d.rung}</td><td class="num">${d.steps/1000}k</td><td${d.pool==='clean'?'':' class="mut"'}>${d.pool}</td>`
  +`<td class="num">${f(d.x_yamnet200,2)}</td><td class="num">${f(d.headline)}</td><td class="num">${f(d.lost_pct,1)}</td><td>${fr.has(d)?'frontier':''}</td></tr>`).join('')+'</tbody>';
 $('list').querySelectorAll('th[data-k]').forEach(th=>th.onclick=()=>{const k=th.dataset.k;sortDir=k===sortKey?-sortDir:(TEXT_COLS.has(k)?1:-1);sortKey=k;draw()});
 $('list').querySelectorAll('tbody tr').forEach(tr=>tr.onclick=()=>pin(DATA[+tr.dataset.i]));
}
function glyph(kind,col='#888',o={}){const sv=document.createElementNS(NS,'svg');sv.setAttribute('viewBox','0 0 22 22');sv.setAttribute('width','15');sv.setAttribute('height','15');
 sv.style.display='inline-block';sv.style.width='15px';sv.style.overflow='visible';sv.appendChild(shape(kind,11,11,col,o));return sv}
function checks(id,key,items,label,deco){
 const el=$(id);el.innerHTML='';const col=FILTERS[key];let fam=null;
 for(const v of items){
  if(key==='fe'){const fm=family(v)[0];if(fm!==fam&&items.filter(i=>family(i)[0]===fm).length>1){const h=document.createElement('div');h.className='fam';h.textContent=fm;el.appendChild(h)}fam=fm}
  const l=document.createElement('label');const c=document.createElement('input');c.type='checkbox';c.checked=state[key].has(v);
  c.onchange=()=>{c.checked?state[key].add(v):state[key].delete(v);draw();detail()};
  l.appendChild(c);if(deco)l.appendChild(deco(v));
  l.appendChild(document.createTextNode(' '+label(v)+' ('+DATA.filter(d=>d[col]===v).length+')'));el.appendChild(l)}
}
function legend(){
 const el=$('legend'),item=(g,t)=>{const s=document.createElement('span');s.appendChild(g);s.appendChild(document.createTextNode(t));el.appendChild(s)};
 for(const r of ['A','B','C'].filter(r=>uniq('rung').includes(r)))item(glyph('circle','#bbb',{rung:r}),'rung '+r);
 item(glyph('circle','#bbb',{faded:true}),'quarantined pool');
 if(DATA.some(d=>d.pre_mmt))item(glyph('circle','#bbb',{slow:true}),'speed timed before mmt (slow)');
 item(glyph('circle','#bbb',{sub:'buzz'}),'buzz only');item(glyph('circle','#bbb',{sub:'all'}),'all 15 classes');
}
const swatch=w=>{const s=document.createElement('span');s.className='sw';s.style.background=WIDTH_COLOUR[w];return s};
function buildChecks(){
 checks('f-w','w',widths,w=>'a'+w,swatch);
 checks('f-d','d',uniq('depth').sort((a,b)=>a-b),d=>'d'+d);
 const fes=uniq('frontend').sort((a,b)=>FAMILIES.indexOf(family(a))-FAMILIES.indexOf(family(b))||(a<b?-1:1));
 checks('f-fe','fe',fes,v=>v,v=>glyph(family(v)[2]));
 checks('f-sub','sub',['sub','buzz','all'].filter(a=>uniq('subset').includes(a)),a=>SUB_NAME[a]);
 checks('f-rung','rung',uniq('rung').sort(),r=>r);
 checks('f-pool','pool',uniq('pool').sort((a,b)=>(b==='clean')-(a==='clean')),p=>p);
}
function init(){
 buildChecks();legend();
 const mx=$('minx'),my=$('miny');mx.min=mx.value=Math.floor(Math.min(...xs)*20)/20;mx.max=Math.max(...xs);
 my.min=my.value=Math.floor(Math.min(...ys)*100)/100;my.max=Math.max(...ys);
 const sync=()=>{$('vx').textContent=(+mx.value).toFixed(2)+'×';$('vy').textContent=(+my.value).toFixed(2)};
 for(const e of [mx,my])e.oninput=()=>{sync();draw()};sync();
 $('q').oninput=e=>{state.q=e.target.value.toLowerCase();draw()};
 for(const id of ['fr-vis','fr-all','labels','join','dim'])$(id).onchange=draw;
 $('reset').onclick=()=>{resetSets();state.q='';$('q').value='';mx.value=mx.min;my.value=my.min;sync();buildChecks();draw()};
 draw();
}
init();
</script></body></html>
"""


def main():
    data = rows()
    html = (PAGE.replace('__DATA__', json.dumps(data, allow_nan=True)).replace('__REFS__', json.dumps(refs()))
            .replace('__TEACHER__', json.dumps(D.TEACHER)))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontier.html')
    open(out, 'w').write(html)
    print(out, len(data), 'runs')


if __name__ == '__main__':
    main()
