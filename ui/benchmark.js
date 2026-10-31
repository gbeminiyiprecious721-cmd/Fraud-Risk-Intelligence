/* Stage 20: immutable aggregate comparison; no model or record endpoints. */
(() => {
  'use strict';
  const el = id => document.getElementById(id);
  const pct = x => x === null ? 'Undefined' : (100*x).toFixed(2)+'%';
  const colors = ['#5aa9ff','#6ed8b5','#e0a93e','#bda2ff'];
  let data = null, selected = 2, loading = false, metricChart = null, capacityChart = null;
  function metric(model) {
    const key = el('benchmark-metric').value;
    return key === 'top1' ? model.metrics.top1.precision : key === 'top5' ? model.metrics.top5.recall : model.metrics[key];
  }
  function select(index) {
    selected = Math.max(0, Math.min(200, Math.round(index)));
    const k = data.models[0].metrics.points[selected].k;
    el('benchmark-slider').value = selected;
    el('benchmark-slider').setAttribute('aria-valuetext', `${k} reviews, ${pct(k/data.rows)}`);
    el('benchmark-count').value = k;
    el('benchmark-percent').textContent = pct(k/data.rows);
    el('benchmark-table').replaceChildren(...data.models.map(model => {
      const point = model.metrics.points[selected], ci = model.uncertainty.ap_ci;
      const row = document.createElement('tr');
      const peak = model.resources.score.rss_peak_bytes / 1024**3;
      const values = [model.name,model.metrics.average_precision.toFixed(5)+(ci ? ` [${ci.map(x=>x.toFixed(4)).join(', ')}]` : ' [unavailable]'),
        `${pct(point.precision)} / ${pct(point.recall)}`,`${point.tp} / ${point.fp} / ${point.fn} / ${point.tn}`,
        model.training_seconds === null ? 'N/A · frozen' : model.training_seconds.toFixed(2),
        `${model.scoring.median_seconds.toFixed(3)}s / ${peak.toFixed(2)} GiB`];
      for (const value of values) {const cell = document.createElement('td');cell.textContent=value;row.append(cell);} return row;
    }));
    if (capacityChart) {
      capacityChart.data.datasets[4].data = [{x:100*k/data.rows,y:0},{x:100*k/data.rows,y:100}];
      capacityChart.update('none');
    }
  }
  function charts() {
    if (typeof Chart === 'undefined') {el('benchmark-status').textContent='Charts unavailable; exact measured results remain in the table.';return;}
    const shared = {responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{labels:{color:'#a9adb8',boxWidth:10,filter:item=>item.text!=='Selected capacity'}}},scales:{y:{min:0,max:1,ticks:{color:'#a9adb8'},grid:{color:'#30343f'}},x:{ticks:{color:'#a9adb8'},grid:{color:'#30343f'}}}};
    metricChart = new Chart(el('benchmark-metric-chart'),{type:'bar',data:{labels:data.models.map(m=>m.name),datasets:[{label:el('benchmark-metric').selectedOptions[0].text,data:data.models.map(metric),backgroundColor:colors}]},options:shared});
    capacityChart = new Chart(el('benchmark-capacity-chart'),{type:'line',data:{datasets:[...data.models.map((m,i)=>({label:m.name,data:m.metrics.points.map(p=>({x:100*p.capacity_fraction,y:100*p.recall})),borderColor:colors[i],pointRadius:0,borderWidth:2,tension:0})),{label:'Selected capacity',data:[],borderColor:'#fff',borderDash:[3,3],pointRadius:0,borderWidth:1}]},options:{...shared,scales:{x:{type:'linear',min:0,max:100,title:{display:true,text:'Workload %',color:'#a9adb8'},ticks:{color:'#a9adb8'},grid:{color:'#30343f'}},y:{min:0,max:100,title:{display:true,text:'Recall %',color:'#a9adb8'},ticks:{color:'#a9adb8'},grid:{color:'#30343f'}}},onClick:(event,elements)=>{if(elements.length && elements[0].datasetIndex<4) select(elements[0].index);}}});
  }
  async function show() {
    if (data) {metricChart?.resize();capacityChart?.resize();return;}
    if (loading) return;
    loading=true;el('benchmark-retry').hidden=true;el('benchmark-status').textContent='Loading measured benchmark evidence…';
    const controller = new AbortController(), timeout=setTimeout(()=>controller.abort(),8000);
    try {
      const local=['localhost','127.0.0.1'].includes(location.hostname)&&location.port==='8080';
      const response=await fetch((local?'http://127.0.0.1:8000':location.origin)+'/v1/dashboard/benchmark',{signal:controller.signal});
      if (!response.ok) throw new Error('Evidence unavailable');
      const body=await response.json();
      if (body.schema_version!==1 || body.rows!==75560 || body.models?.length!==4 || !body.models.every(m=>m.metrics.points.length===201 && m.metrics.points[0].k===0 && m.metrics.points.at(-1).k===body.rows)) throw new Error('Invalid evidence');
      data=body;el('benchmark-workspace').hidden=false;el('benchmark-count').max=body.rows;
      el('benchmark-status').textContent='';
      el('benchmark-population').textContent=`${body.rows.toLocaleString()} validation rows · ${body.positives.toLocaleString()} verified positives · 813 shared features · 1,000 paired block draws · experimental challengers`;
      el('benchmark-finding').textContent=`Highest observed AP: ${body.strongest_observed} within this budget · findings & limitations`;
      el('benchmark-limits').replaceChildren(...body.limitations.map(text=>{const p=document.createElement('p');p.textContent=text;return p;}));
      const differences=document.createElement('p');differences.textContent='Paired AP differences vs RF, descriptive 95% block intervals: '+body.models.slice(1).map(m=>m.name+' '+(m.uncertainty.ap_difference_ci?.map(x=>x.toFixed(5)).join(' to ') || 'unavailable')).join(' · ');el('benchmark-limits').append(differences);
      el('benchmark-kpis').replaceChildren(...body.models.map(m=>{const card=document.createElement('div');card.className='kpi-card';const label=document.createElement('p');label.className='kpi-label';label.textContent=m.name+' · '+(m.training===null?'deployed':'experimental');const value=document.createElement('p');value.className='kpi-value accent';value.textContent=m.metrics.average_precision.toFixed(5)+' AP';value.title='Average Precision';const note=document.createElement('span');note.className='panel-note';note.textContent='Average Precision';card.append(label,value,note);return card;}));
      charts();select(2);
    } catch(error) {data=null;el('benchmark-workspace').hidden=true;el('benchmark-status').textContent='Measured benchmark evidence unavailable. No comparison is shown.';el('benchmark-retry').hidden=false;}
    finally {clearTimeout(timeout);loading=false;}
  }
  el('benchmark-metric').addEventListener('change',()=>{if(metricChart){metricChart.data.datasets[0].data=data.models.map(metric);metricChart.data.datasets[0].label=el('benchmark-metric').selectedOptions[0].text;metricChart.update('none');}});
  el('benchmark-slider').addEventListener('input',event=>{if(data)select(Number(event.target.value));});
  el('benchmark-count').addEventListener('change',event=>{if(!data)return;const n=event.target.valueAsNumber;if(!Number.isFinite(n)){select(selected);return;}const points=data.models[0].metrics.points;select(points.reduce((best,p,i)=>Math.abs(p.k-n)<Math.abs(points[best].k-n)?i:best,0));});
  el('benchmark-reset').addEventListener('click',()=>{if(data){el('benchmark-metric').value='average_precision';el('benchmark-metric').dispatchEvent(new Event('change'));select(2);}});
  el('benchmark-retry').addEventListener('click',show);
  window.ModelBenchmark={show};
})();
