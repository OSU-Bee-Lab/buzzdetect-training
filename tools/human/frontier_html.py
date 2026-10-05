"""Interactive frontier page: the frontier_svg.py scatter as one self-contained HTML file (no network, no libraries).

Hover a point for its exact model name and numbers; click to pin it and see per-fold sensitivity, loudness tiers and the
jet-fold probe. Filters: trunk, front end, class subset, speed/sensitivity range, name search; the dashed frontier can be
computed over the visible points or over every run.

    python tools/human/frontier_html.py [out.html]      (train env; frontier_svg.load_all: every rung, clean and tagged pools)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '05_distill'))
import dpaths as D  # noqa: E402
import frontier_svg as F  # noqa: E402

KEEP = ('name', 'frontend', 'arch', 'classes', 'init', 'seed', 'steps', 'headline', 'headline_inclusive', 'x_yamnet200',
        'x_yamnet20', 'lost_pct', 'gained_pct', 'mae_live', 'mae_buzz', 'per_fold', 'tiers', 'wall_s', 'rung', 'pool')


def rows():
    out = []
    for r in F.load_all():            # every rung, one point per student and rung (frontier_svg.py)
        d = {k: r.get(k) for k in KEEP}
        d['subset'] = F.subset(r) or 'all'
        d['probe'] = F.probe_of(r['name'])
        out.append(d)
    return out


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Distillation frontier</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--grid:#e5e5e5;--box:#999;--panel:#f6f6f6;--hi:#111}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e8e8e8;--mut:#999;--grid:#2c2c2c;--box:#666;--panel:#222;--hi:#fff}}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.4 system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px}
h1{font-size:16px;margin:0 0 2px}p.sub{margin:0 0 12px;color:var(--mut)}
#layout{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
#controls{flex:0 0 230px;display:flex;flex-direction:column;gap:12px}
fieldset{border:1px solid var(--grid);border-radius:6px;margin:0;padding:6px 10px 8px}
legend{font-weight:600;padding:0 4px}
label{display:flex;align-items:center;gap:6px;cursor:pointer;padding:1px 0}
.sw{width:11px;height:11px;border-radius:2px;display:inline-block}
input[type=search]{width:100%;box-sizing:border-box;padding:4px;background:var(--panel);color:var(--fg);border:1px solid var(--grid);border-radius:4px}
input[type=range]{width:100%}
button{background:var(--panel);color:var(--fg);border:1px solid var(--box);border-radius:4px;padding:3px 8px;cursor:pointer}
#chartwrap{flex:1 1 640px;min-width:0;position:relative}
svg{width:100%;height:auto;display:block}svg text{fill:var(--fg)}
#tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--box);border-radius:6px;padding:6px 8px;
 font-size:12px;display:none;max-width:340px;box-shadow:0 2px 8px #0004;z-index:2}
#tip b{word-break:break-all}
#detail{margin-top:12px;background:var(--panel);border-radius:6px;padding:8px 12px;min-height:40px}
#detail table{border-collapse:collapse;margin-top:4px}#detail td,#detail th{padding:1px 10px 1px 0;text-align:left;font-weight:normal}
#detail th{color:var(--mut)}
table.list{border-collapse:collapse;width:100%;margin-top:12px}
table.list th{cursor:pointer;text-align:left;border-bottom:1px solid var(--box);padding:3px 6px;white-space:nowrap}
table.list td{padding:2px 6px;border-bottom:1px solid var(--grid)}
table.list tr:hover td{background:var(--panel)}
.num{text-align:right;font-variant-numeric:tabular-nums}
.mut{color:var(--mut)}
</style></head><body><main>
<h1>Speed / sensitivity frontier (every rung, seed 1, each student at its stop-rule budget)</h1>
<p class="sub" id="sub"></p>
<div id="layout">
 <div id="controls">
  <fieldset><legend>Trunk</legend><div id="f-arch"></div></fieldset>
  <fieldset><legend>Front end</legend><div id="f-fe"></div></fieldset>
  <fieldset><legend>Classes</legend><div id="f-sub"></div></fieldset>
  <fieldset><legend>Search</legend><input type="search" id="q" placeholder="model name contains…"></fieldset>
  <fieldset><legend>Min speed: <span id="vx"></span></legend><input type="range" id="minx" step="0.05"></fieldset>
  <fieldset><legend>Min headline: <span id="vy"></span></legend><input type="range" id="miny" step="0.01"></fieldset>
  <fieldset><legend>Display</legend>
   <label><input type="checkbox" id="fr-vis" checked> frontier of visible points</label>
   <label><input type="checkbox" id="fr-all"> also show the frontier of all runs</label>
   <label><input type="checkbox" id="labels" checked> label frontier points</label>
   <label><input type="checkbox" id="dim"> hide non-frontier points</label>
  </fieldset>
  <button id="reset">Reset filters</button>
 </div>
 <div id="chartwrap"><svg id="chart" viewBox="0 0 760 520"></svg><div id="tip"></div></div>
</div>
<div id="detail"><span class="mut">Click a point (or a table row) to pin its details.</span></div>
<table class="list" id="list"></table>
</main>
<script>
const DATA=__DATA__, BASELINE=__BASELINE__;
const ARCH_COLOUR={'a0.25':'#d95f02','a0.375':'#7570b3','a0.50':'#1b9e77','a0.50_d12':'#e7298a'};
const FE_MARK={yamnet:'circle',fast32:'square',fast32h16:'diamond',fast32h32:'tri',twofast32:'cross',two32:'hex',lo32:'hex',fast32lo:'square',fast32h16lo:'diamond'};
const SUB_NAME={all:'all 15 classes',sub:'buzz + rain + human',buzz:'buzz only'};
const SUB_TAG={all:'',sub:' +3cls',buzz:' +buzz'};
const W=760,H=520,L=62,R=14,T=14,B=48;
const $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
const uniq=k=>[...new Set(DATA.map(d=>d[k]))];
const state={arch:new Set(uniq('arch')),fe:new Set(uniq('frontend')),sub:new Set(uniq('subset')),q:'',pinned:null};
const xs=DATA.map(d=>d.x_yamnet200), ys=DATA.map(d=>d.headline);
const X0=Math.floor(Math.min(...xs)*10)/10-0.1, X1=Math.max(...xs)+0.2, Y0=0.35, Y1=0.75;
const px=v=>L+(v-X0)/(X1-X0)*(W-L-R), py=v=>H-B-(v-Y0)/(Y1-Y0)*(H-B-T);

function shape(kind,x,y,col,sub){
 const s=7,g=document.createElementNS(NS,'g');let e;
 const mk=(t,a)=>{const n=document.createElementNS(NS,t);for(const k in a)n.setAttribute(k,a[k]);return n};
 const poly=p=>mk('polygon',{points:p.map(q=>q.join(',')).join(' ')});
 if(kind==='square')e=mk('rect',{x:x-s,y:y-s,width:2*s,height:2*s});
 else if(kind==='diamond')e=poly([[x,y-s-2],[x+s+2,y],[x,y+s+2],[x-s-2,y]]);
 else if(kind==='tri')e=poly([[x,y-s-1],[x+s+1,y+s],[x-s-1,y+s]]);
 else if(kind==='hex')e=poly([[x-s,y],[x-s/2,y-s],[x+s/2,y-s],[x+s,y],[x+s/2,y+s],[x-s/2,y+s]]);
 else if(kind==='cross'){const a=2.5;e=poly([[x-a,y-s],[x+a,y-s],[x+a,y-a],[x+s,y-a],[x+s,y+a],[x+a,y+a],[x+a,y+s],[x-a,y+s],[x-a,y+a],[x-s,y+a],[x-s,y-a],[x-a,y-a]])}
 else e=mk('circle',{cx:x,cy:y,r:s});
 e.setAttribute('fill',col);e.setAttribute('fill-opacity','.85');
 if(sub!=='all'){e.setAttribute('stroke','var(--hi)');e.setAttribute('stroke-width','2.5')}
 g.appendChild(e);
 if(sub==='buzz')g.appendChild(mk('circle',{cx:x,cy:y,r:2.2,fill:'var(--hi)'}));
 return g;
}
function pareto(rows){
 let best=-1;const out=new Set();
 for(const r of [...rows].sort((a,b)=>b.x_yamnet200-a.x_yamnet200))if(r.headline>best){out.add(r);best=r.headline}
 return out;
}
const f=(v,d=3)=>v==null||v!==v?'–':(+v).toFixed(d);
function visible(){
 return DATA.filter(d=>state.arch.has(d.arch)&&state.fe.has(d.frontend)&&state.sub.has(d.subset)
  &&d.name.toLowerCase().includes(state.q)&&d.x_yamnet200>=+$('minx').value&&d.headline>=+$('miny').value);
}
function tipHtml(d){
 return `<b>${d.name}</b><br>${d.frontend} · ${d.arch} · ${SUB_NAME[d.subset]}${d.init?' · init '+d.init:''}`
  +`<br>headline <b>${f(d.headline)}</b> (${f(d.headline/BASELINE*100,0)}% of baseline)`
  +`<br>speed <b>${f(d.x_yamnet200,2)}×</b> YAMNet · lost buzz ${f(d.lost_pct,1)}%`;
}
const tip=$('tip');
function showTip(ev,d){
 const wr=$('chartwrap').getBoundingClientRect();tip.innerHTML=tipHtml(d);tip.style.display='block';
 let x=ev.clientX-wr.left+14,y=ev.clientY-wr.top+14;
 if(x+tip.offsetWidth>wr.width)x=ev.clientX-wr.left-tip.offsetWidth-14;
 tip.style.left=x+'px';tip.style.top=y+'px';
}
function pin(d){state.pinned=d;draw();detail()}
function detail(){
 const d=state.pinned,el=$('detail');
 if(!d){el.innerHTML='<span class="mut">Click a point (or a table row) to pin its details.</span>';return}
 const pf=(d.per_fold||[]).map(v=>f(v,2)).join(' · ');
 const tiers=Object.entries(d.tiers||{}).filter(([,v])=>v!=null).map(([k,v])=>`${k} ${f(v,2)}`).join(' · ');
 const p=d.probe;
 el.innerHTML=`<b>${d.name}</b> <span class="mut">${d.frontend} · ${d.arch} · ${SUB_NAME[d.subset]} · rung ${d.rung} · ${d.pool} · seed ${d.seed} · ${d.steps} steps · wall ${f(d.wall_s/60,0)} min</span>
 <table><tr><th>headline (excl. quiet, fpr 0.005)</th><td>${f(d.headline)}</td><th>incl. quiet</th><td>${f(d.headline_inclusive)}</td></tr>
 <tr><th>speed (200 s / 20 s)</th><td>${f(d.x_yamnet200,2)}× / ${f(d.x_yamnet20,2)}×</td><th>buzz lost / gained</th><td>${f(d.lost_pct,1)}% / ${f(d.gained_pct,1)}%</td></tr>
 <tr><th>logit error (all / buzz)</th><td>${f(d.mae_live)} / ${f(d.mae_buzz)}</td><th></th><td></td></tr>
 <tr><th>per rotating fold</th><td colspan="3">${pf}</td></tr>
 <tr><th>by loudness tier</th><td colspan="3">${tiers}</td></tr>
 <tr><th>jet fold 1_95 FPR</th><td colspan="3">${p&&p.pooled?`all negatives ${f(p.pooled.fpr*100,2)}% · jet frames ${f(p.pooled.fpr_jet*100,2)}% · worst rotating fold ${f(p.pooled.rotating_fpr_max*100,2)}%`:'<span class="mut">no probe.json</span>'}</td></tr></table>`;
}
function draw(){
 const svg=$('chart'),vis=visible(),visSet=new Set(vis);svg.innerHTML='';
 const add=(t,a,txt)=>{const n=document.createElementNS(NS,t);for(const k in a)n.setAttribute(k,a[k]);if(txt!=null)n.textContent=txt;svg.appendChild(n);return n};
 for(const v of [0.4,0.5,0.6,0.7]){add('line',{x1:L,x2:W-R,y1:py(v),y2:py(v),stroke:'var(--grid)'});add('text',{x:L-8,y:py(v)+4,'text-anchor':'end'},v.toFixed(1))}
 for(let v=Math.ceil(X0*2)/2;v<X1;v+=0.5){add('line',{x1:px(v),x2:px(v),y1:T,y2:H-B,stroke:'var(--grid)'});add('text',{x:px(v),y:H-B+16,'text-anchor':'middle'},v+'×')}
 if(X0<1&&X1>1){add('line',{x1:px(1),x2:px(1),y1:T,y2:H-B,stroke:'var(--mut)','stroke-dasharray':'4 3'});add('text',{x:px(1)+4,y:T+12,fill:'var(--mut)'},'YAMNet speed')}
 add('line',{x1:L,x2:W-R,y1:py(BASELINE),y2:py(BASELINE),stroke:'var(--mut)','stroke-dasharray':'4 3'});
 add('text',{x:W-R-4,y:py(BASELINE)-5,'text-anchor':'end',fill:'var(--mut)'},'era baseline '+BASELINE);
 add('rect',{x:L,y:T,width:W-L-R,height:H-B-T,fill:'none',stroke:'var(--box)'});
 add('text',{x:(L+W-R)/2,y:H-10,'text-anchor':'middle'},'speed, × YAMNet (GPU, 200 s audio; higher is faster)');
 add('text',{transform:`translate(16 ${(T+H-B)/2}) rotate(-90)`,'text-anchor':'middle'},'sensitivity (excl. quiet) @ 0.5% FPR');
 const frAll=pareto(DATA),frVis=pareto(vis);
 const line=(set,dash,col)=>{const p=[...set].filter(r=>visSet.has(r)||set===frAll).sort((a,b)=>b.x_yamnet200-a.x_yamnet200);
  if(p.length)add('polyline',{fill:'none',stroke:col,'stroke-width':1.5,'stroke-dasharray':dash,points:p.map(r=>px(r.x_yamnet200).toFixed(1)+','+py(r.headline).toFixed(1)).join(' ')})};
 if($('fr-all').checked)line(frAll,'1 4','var(--mut)');
 if($('fr-vis').checked)line(frVis,'2 3','var(--hi)');

 const frList=[...frVis].sort((a,b)=>b.x_yamnet200-a.x_yamnet200);
 for(const d of vis){
  const isFr=frVis.has(d);if($('dim').checked&&!isFr)continue;
  const g=shape(FE_MARK[d.frontend]||'circle',+px(d.x_yamnet200).toFixed(1),+py(d.headline).toFixed(1),ARCH_COLOUR[d.arch]||'#888',d.subset);
  g.style.cursor='pointer';if(!isFr&&$('fr-vis').checked)g.style.opacity=.75;
  g.addEventListener('mousemove',e=>showTip(e,d));g.addEventListener('mouseleave',()=>tip.style.display='none');
  g.addEventListener('click',()=>pin(d));svg.appendChild(g);
  if(isFr&&$('labels').checked){const k=frList.indexOf(d),right=px(d.x_yamnet200)>W-130;
   add('text',{x:px(d.x_yamnet200)+(right?-10:10),y:py(d.headline)-9-(k%2)*12,'font-size':10,'text-anchor':right?'end':'start'},d.frontend+SUB_TAG[d.subset])}
 }
 if(state.pinned&&visSet.has(state.pinned)){const d=state.pinned;
  add('circle',{cx:px(d.x_yamnet200),cy:py(d.headline),r:13,fill:'none',stroke:'var(--hi)','stroke-width':1.5,'stroke-dasharray':'3 2'})}
 $('sub').textContent=`${vis.length} of ${DATA.length} runs shown · ${frVis.size} on the frontier · hover for the model name, click to pin`;
 table(vis,frVis);
}
let sortKey='headline',sortDir=-1;
function table(vis,fr){
 const cols=[['name','model'],['subset','classes'],['arch','trunk'],['frontend','front end'],['x_yamnet200','speed ×'],['headline','headline'],['lost_pct','lost %']];
 const rows=[...vis].sort((a,b)=>{const x=a[sortKey],y=b[sortKey];return (x<y?-1:x>y?1:0)*sortDir});
 $('list').innerHTML='<thead><tr>'+cols.map(([k,t])=>`<th data-k="${k}">${t}${k===sortKey?(sortDir<0?' ▼':' ▲'):''}</th>`).join('')+'<th></th></tr></thead><tbody>'
  +rows.map((d,i)=>`<tr data-i="${DATA.indexOf(d)}"><td>${d.name}</td><td>${SUB_NAME[d.subset]}</td><td>${d.arch}</td><td>${d.frontend}</td>`
  +`<td class="num">${f(d.x_yamnet200,2)}</td><td class="num">${f(d.headline)}</td><td class="num">${f(d.lost_pct,1)}</td><td>${fr.has(d)?'frontier':''}</td></tr>`).join('')+'</tbody>';
 $('list').querySelectorAll('th[data-k]').forEach(th=>th.onclick=()=>{const k=th.dataset.k;sortDir=k===sortKey?-sortDir:(k==='name'||k==='arch'||k==='frontend'||k==='subset'?1:-1);sortKey=k;draw()});
 $('list').querySelectorAll('tbody tr').forEach(tr=>tr.onclick=()=>pin(DATA[+tr.dataset.i]));
}
function glyph(fe){const sv=document.createElementNS(NS,'svg');sv.setAttribute('viewBox','0 0 20 20');sv.setAttribute('width','14');sv.setAttribute('height','14');
 sv.style.display='inline-block';sv.style.width='14px';sv.appendChild(shape(FE_MARK[fe]||'circle',10,10,'#888','all'));return sv}
function checks(id,key,items,label,swatch,glyphs){
 const el=$(id);el.innerHTML='';
 for(const v of items){const l=document.createElement('label');const c=document.createElement('input');c.type='checkbox';c.checked=state[key].has(v);
  c.onchange=()=>{c.checked?state[key].add(v):state[key].delete(v);draw();detail()};
  l.appendChild(c);if(glyphs)l.appendChild(glyph(v));if(swatch){const s=document.createElement('span');s.className='sw';s.style.background=swatch(v);l.appendChild(s)}
  l.appendChild(document.createTextNode(' '+label(v)+' ('+DATA.filter(d=>d[key==='fe'?'frontend':key==='sub'?'subset':key]===v).length+')'));el.appendChild(l)}
}
function init(){
 const order=['a0.25','a0.375','a0.50','a0.50_d12'].filter(a=>uniq('arch').includes(a));
 checks('f-arch','arch',order,a=>a,a=>ARCH_COLOUR[a]);
 checks('f-fe','fe',uniq('frontend').sort(),a=>a,null,true);
 checks('f-sub','sub',['all','sub','buzz'].filter(a=>uniq('subset').includes(a)),a=>SUB_NAME[a]);
 const mx=$('minx'),my=$('miny');mx.min=mx.value=Math.floor(Math.min(...xs)*20)/20;mx.max=Math.max(...xs);
 my.min=my.value=Math.floor(Math.min(...ys)*100)/100;my.max=Math.max(...ys);
 const sync=()=>{$('vx').textContent=(+mx.value).toFixed(2)+'×';$('vy').textContent=(+my.value).toFixed(2)};
 for(const e of [mx,my])e.oninput=()=>{sync();draw()};sync();
 $('q').oninput=e=>{state.q=e.target.value.toLowerCase();draw()};
 for(const id of ['fr-vis','fr-all','labels','dim'])$(id).onchange=draw;
 $('reset').onclick=()=>{for(const k of ['arch','fe','sub'])state[k]=new Set(k==='fe'?uniq('frontend'):k==='sub'?uniq('subset'):uniq('arch'));
  state.q='';$('q').value='';mx.value=mx.min;my.value=my.min;sync();
  checks('f-arch','arch',order,a=>a,a=>ARCH_COLOUR[a]);checks('f-fe','fe',uniq('frontend').sort(),a=>a,null,true);
  checks('f-sub','sub',['all','sub','buzz'].filter(a=>uniq('subset').includes(a)),a=>SUB_NAME[a]);draw()};
 draw();
}
init();
</script></body></html>
"""


def main():
    data = rows()
    html = PAGE.replace('__DATA__', json.dumps(data, allow_nan=True)).replace('__BASELINE__', str(F.BASELINE))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontier.html')
    open(out, 'w').write(html)
    print(out, len(data), 'runs')


if __name__ == '__main__':
    main()
