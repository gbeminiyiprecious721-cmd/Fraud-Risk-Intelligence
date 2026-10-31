/* Historical aggregate-only monitoring. No model calls or local case-store writes. */
(() => {
  'use strict';
  const el=id=>document.getElementById(id);
  const names={TransactionAmt:'Transaction amount',ProductCD:'Product code',card4:'Card network',C1:'C1 (anonymised count)',dist1:'dist1 (units unverified)',daily_phase:'Relative daily phase'};
  const pct=v=>v===null?'Undefined':(100*v).toFixed(2)+'%';
  const share=(n,d)=>d>0?n/d:null;
  const num=v=>v===null?'Undefined':Number(v).toFixed(4);
  const colors=['#5aa9ff','#6ed8b5'];
  let data=null,loading=false,main=null,detail=null;
  const opts={responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{labels:{color:'#a9adb8',boxWidth:10}}},scales:{x:{ticks:{color:'#a9adb8',maxRotation:20,font:{size:10}},grid:{color:'#30343f'}},y:{beginAtZero:true,ticks:{color:'#a9adb8'},grid:{color:'#30343f'}}}};
  function chart(id,type,labels,datasets,extra={}) {
    if(typeof Chart==='undefined')return null;
    return new Chart(el(id),{type,data:{labels,datasets},options:{...opts,...extra}});
  }
  function series(label,values,i=0){return {label,data:values,backgroundColor:colors[i],borderColor:colors[i],borderWidth:2,pointRadius:3,tension:0};}
  function ratio(r){return r.kind==='numeric'?r.psi===null?null:Math.max(r.psi/.2,r.ks/.1):r.js===null?null:r.js/.05;}
  function qualitySignal(r){return r.missing_delta>=.05||r.comparison_quality.parse_failures>0||r.comparison_quality.nonfinite>0;}
  function row(values){const tr=document.createElement('tr');for(const v of values){const td=document.createElement('td');td.textContent=v;tr.append(td);}return tr;}
  function render() {
    if(!data)return;
    const windowId=el('monitoring-window').value,feature=el('monitoring-feature').value,mode=el('monitoring-mode').value;
    const c=data.comparisons.find(x=>x.window===windowId),w=data.windows.find(x=>x.id===windowId),ref=data.windows[0];
    if(!c||!names[feature]||!['features','scores','quality'].includes(mode)){el('monitoring-status').textContent='Unsupported monitoring selection. Reset monitoring filters to recover.';return;}
    el('monitoring-feature').disabled=mode==='scores';
    el('monitoring-status').textContent=typeof Chart==='undefined'?'Charts unavailable; exact aggregate values remain below.':'';
    el('monitoring-population').textContent=`Reference V1: ${ref.n.toLocaleString()} rows, relative seconds ${ref.start}–${ref.end} · ${w.id}: ${w.n.toLocaleString()} rows, ${w.start}–${w.end}. No overlap; no labels used. Synthetic batch filters do not apply.`;
    const entries=Object.entries(c.features),shifts=entries.filter(([,r])=>r.signal==='Investigate shift').length;
    const qualityIssues=entries.filter(([,r])=>qualitySignal(r)).length+(w.negative_amounts>0?1:0)+(w.duplicate_ids>0?1:0);
    const cards=[['Feature shift signals',String(shifts)+' / '+entries.length],['Quality investigation signals',String(qualityIssues)],['Score shift signal',c.score.signal==='Investigate shift'?'Investigate':c.score.signal==='Insufficient evidence'?'Insufficient':'Below rules'],['Above fixed reference q95',pct(w.tail_fraction)]];
    el('monitoring-kpis').replaceChildren(...cards.map(([label,value])=>{const d=document.createElement('div');d.className='kpi-card';const p=document.createElement('p');p.className='kpi-label';p.textContent=label;const v=document.createElement('p');v.className='kpi-value accent';v.textContent=value;d.append(p,v);return d;}));
    main?.destroy();detail?.destroy();main=null;detail=null;
    const r=mode==='scores'?c.score:c.features[feature];
    el('monitoring-column').textContent=w.id;
    const q=r.comparison_quality,rq=r.reference_quality;
    let text=mode==='scores'?'Uncalibrated score':names[feature];
    text+=` · ${r.signal}. `+(r.kind==='numeric'?`PSI ${num(r.psi)}, KS ${num(r.ks)}.`:`JS divergence ${num(r.js)}; reference-unseen count ${r.unseen_count}.`);
    text+=` Missing ${rq.missing}/${rq.n} → ${q.missing}/${q.n} (${(100*r.missing_delta).toFixed(2)} pp change). Invalid numeric: parse failures ${q.parse_failures}, nonfinite ${q.nonfinite}.`;
    if(qualitySignal(r))text+=' Data-quality investigation warranted under selected rules.';
    if(mode==='scores')text+=` Fixed reference q95=${num(data.reference_tail_threshold)}; strict-tail fraction ${pct(ref.tail_fraction)} → ${pct(w.tail_fraction)}. Drift does not establish performance deterioration.`;
    if(mode==='quality')text+=` Window schema checked; duplicate IDs ${w.duplicate_ids}; negative amounts ${w.negative_amounts}. Reference-unseen categories are not automatically invalid.`;
    if(r.signal==='Insufficient evidence')text+=' Distribution metrics unavailable: fewer than 200 eligible observations. Undefined chart shares are omitted.';
    el('monitoring-detail').textContent=text;
    if(mode==='features'){
      const ranked=entries.slice().sort((a,b)=>(ratio(b[1])??-1)-(ratio(a[1])??-1));
      el('monitoring-chart-title').textContent='Feature investigation-rule ratios';
      el('monitoring-chart-note').textContent='Indicator / fixed heuristic. ≥1: investigate; not significance or feature importance.';
      main=chart('monitoring-main-chart','bar',ranked.map(([k])=>names[k]),[series('Indicator / demonstration rule',ranked.map(([,v])=>ratio(v)))],{indexAxis:'y',plugins:{legend:{display:false}},scales:{x:opts.scales.x,y:{...opts.scales.y,ticks:{...opts.scales.y.ticks,autoSkip:false,font:{size:10}}}}});
    }else if(mode==='scores'){
      el('monitoring-chart-title').textContent='Frozen Random Forest score distribution';
      el('monitoring-chart-note').textContent='Shared reference-only bins; uncalibrated ranking scores, not fraud probabilities.';
      main=chart('monitoring-main-chart','bar',r.labels,[series('Reference V1 share',r.reference_counts.map(n=>share(n,rq.finite))),series(w.id+' share',r.comparison_counts.map(n=>share(n,q.finite)),1)]);
    }else{
      el('monitoring-chart-title').textContent='Missing-value rates by selected input';
      el('monitoring-chart-note').textContent='All-row denominators. Missingness is separate from finite-value drift; invalid numeric counts and schema checks appear below.';
      main=chart('monitoring-main-chart','bar',entries.map(([k])=>names[k]),[series('Reference V1 missing %',entries.map(([,v])=>100*v.reference_quality.missing_rate)),series(w.id+' missing %',entries.map(([,v])=>100*v.comparison_quality.missing_rate),1)]);
    }
    if(mode==='features'){
      el('monitoring-detail-title').textContent=names[feature]+' · distribution';
      const labels=r.kind==='numeric'?r.labels:r.categories;
      detail=chart('monitoring-detail-chart','bar',labels,[series('Reference V1 share',r.reference_counts.map(n=>share(n,r.kind==='numeric'?rq.finite:rq.n))),series(w.id+' share',r.comparison_counts.map(n=>share(n,r.kind==='numeric'?q.finite:q.n)),1)]);
      let buckets=labels;
      if(r.kind==='numeric'&&r.labels[0]!=='Reference value')buckets=labels.map((label,i)=>label+` [${i===0?'−∞':num(r.edges[i-1])}, ${i===labels.length-1?'+∞':num(r.edges[i])})`);
      el('monitoring-table-caption').textContent='Exact selected distribution counts; numeric missing/nonfinite excluded, categorical missing/unseen included';
      el('monitoring-table').replaceChildren(...buckets.map((label,i)=>row([label,r.reference_counts[i],r.comparison_counts[i]])));
    }else{
      el('monitoring-detail-title').textContent=mode==='scores'?'Historical score tail · three descriptive windows':'Selected feature missingness · three descriptive windows';
      const values=mode==='scores'?data.windows.map(v=>100*v.tail_fraction):[100*c.features[feature].reference_quality.missing_rate,...data.comparisons.map(v=>100*v.features[feature].comparison_quality.missing_rate)];
      detail=chart('monitoring-detail-chart','line',data.windows.map(v=>v.id),[series(mode==='scores'?'Above fixed reference q95 %':'Missing %',values)]);
      if(mode==='scores'){
        el('monitoring-table-caption').textContent='Score quantiles and mean · displayed to 4 decimals; uncalibrated ranks';
        el('monitoring-table').replaceChildren(row(['Mean',num(r.reference_summary.mean),num(r.comparison_summary.mean)]),...['q01','q10','q50','q90','q95','q99'].map((label,i)=>row([label,num(r.reference_summary.quantiles[i]),num(r.comparison_summary.quantiles[i])])));
      }else{
        el('monitoring-table-caption').textContent='Exact selected-feature quality counts';
        el('monitoring-table').replaceChildren(...['n','missing','parse_failures','nonfinite','finite'].map(k=>row([k.replaceAll('_',' '),rq[k],q[k]])),row(['Missing rate',pct(rq.missing_rate),pct(q.missing_rate)]),row(['Duplicate IDs',ref.duplicate_ids,w.duplicate_ids]),row(['Negative amounts',ref.negative_amounts,w.negative_amounts]));
      }
    }
  }
  function reset(){el('monitoring-window').value='V2';el('monitoring-mode').value='features';el('monitoring-feature').value='TransactionAmt';render();}
  async function show(){
    if(data){main?.resize();detail?.resize();return;}if(loading)return;
    loading=true;el('monitoring-retry').hidden=true;el('monitoring-status').textContent='Loading historical monitoring aggregates…';
    const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),8000);
    try{
      const local=['127.0.0.1','localhost'].includes(location.hostname)&&location.port==='8080';
      const response=await fetch((local?'http://127.0.0.1:8000':location.origin)+'/v1/dashboard/monitoring',{signal:controller.signal});
      if(!response.ok)throw new Error('Unavailable');const body=await response.json();
      if(body.schema_version!==1||body.source!=='historical_validation'||body.model_sha256!=='2fb0cbc4457ec2e63ce235d5faee92c2812f3811c9efdf872760268d9122bf04'||body.windows?.length!==3||body.comparisons?.length!==2||!body.comparisons.every(c=>Object.keys(names).every(k=>c.features[k])&&c.score)||!body.limitations?.length)throw new Error('Invalid aggregate');
      data=body;el('monitoring-workspace').hidden=false;
      el('monitoring-limits').replaceChildren(...body.limitations.map(t=>{const p=document.createElement('p');p.textContent=t;return p;}));reset();
    }catch(error){data=null;main?.destroy();detail?.destroy();main=null;detail=null;el('monitoring-workspace').hidden=true;el('monitoring-status').textContent='Historical monitoring evidence unavailable or invalid. No monitoring result is shown.';el('monitoring-retry').hidden=false;}
    finally{clearTimeout(timer);loading=false;}
  }
  for(const id of ['monitoring-window','monitoring-mode','monitoring-feature'])el(id).addEventListener('change',render);
  el('monitoring-reset').addEventListener('click',reset);el('monitoring-retry').addEventListener('click',show);
  window.ModelMonitoring={show};
})();
