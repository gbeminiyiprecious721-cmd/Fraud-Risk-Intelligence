/* Stage 19: fixed historical aggregate only. No per-transaction data or inference. */
(() => {
  'use strict';
  const el = id => document.getElementById(id);
  const fmt = n => Number(n).toLocaleString();
  const pct = n => n === null ? 'Not defined' : (100 * n).toFixed(2) + '%';
  let aggregate = null, selected = 2, loading = false;
  const charts = [];

  function select(index) {
    selected = Math.max(0, Math.min(aggregate.points.length - 1, Math.round(index)));
    const p = aggregate.points[selected], previous = aggregate.points[selected - 1];
    el('capacity-slider').value = selected;
    el('capacity-slider').setAttribute('aria-valuetext', `${fmt(p.k)} transactions, ${pct(p.capacity_fraction)}`);
    el('capacity-count').value = p.k;
    el('capacity-percent').textContent = pct(p.capacity_fraction);
    const cards = [['Reviewed', fmt(p.k)], ['Historical frauds detected', fmt(p.tp)], ['Precision at capacity', pct(p.precision)], ['Recall / fraud capture', pct(p.recall)], ['Unreviewed', fmt(aggregate.n - p.k)]];
    el('capacity-kpis').replaceChildren(...cards.map(([label, value]) => {
      const card = document.createElement('div'); card.className = 'kpi-card';
      const title = document.createElement('p'); title.className = 'kpi-label'; title.textContent = label;
      const number = document.createElement('p'); number.className = 'kpi-value accent'; number.textContent = value;
      card.append(title, number); return card;
    }));
    const extra = previous ? `Additional ${fmt(p.k - previous.k)} reviews vs. the prior checkpoint captured ${fmt(p.tp - previous.tp)} historical frauds; block precision ${pct((p.tp - previous.tp) / (p.k - previous.k))}.` : 'At zero capacity, no transactions are reviewed and precision is undefined.';
    el('capacity-brief').textContent = `Random review expects ${p.random_tp_expected.toFixed(1)} frauds; ${previous ? 'the latest review block yields ' + pct((p.tp - previous.tp) / (p.k - previous.k)) + ' precision.' : 'no reviews selected.'}`;
    el('capacity-decision').textContent = `Uniform random sampling without replacement expects ${p.random_tp_expected.toFixed(1)} frauds. ${extra}`;
    el('capacity-counts').textContent = `True positives ${fmt(p.tp)} · False positives ${fmt(p.fp)} · False negatives ${fmt(p.fn)} · True negatives ${fmt(p.tn)} · Random precision ${pct(p.random_precision_expected)} / recall ${pct(p.random_recall_expected)}.`;
    for (const [i, chart] of charts.entries()) {
      chart.data.datasets[2].data = [{x:p.capacity_fraction * 100, y:(i === 0 ? p.recall : p.precision) === null ? null : (i === 0 ? p.recall : p.precision) * 100}];
      chart.update('none');
    }
  }

  function buildCharts() {
    if (typeof Chart === 'undefined') {
      el('capacity-status').textContent = 'Charts unavailable: the Chart.js library could not load. Capacity inputs and exact metrics remain available.';
      return;
    }
    charts.splice(0).forEach(chart => chart.destroy());
    ['recall', 'precision'].forEach((metric, i) => {
      const points = aggregate.points;
      const values = key => points.map(p => ({x:100 * p.capacity_fraction, y:p[key] === null ? null : 100 * p[key]}));
      const canvas = el(`capacity-${metric}-chart`);
      const chart = new Chart(canvas, {
        type:'line',
        data:{datasets:[
          {label:'Historical validation', data:values(metric), borderColor:'#5aa9ff', backgroundColor:'#5aa9ff', pointRadius:0, pointHitRadius:10, borderWidth:2, tension:0},
          {label:'Random review (expected)', data:values(`random_${metric}_expected`), borderColor:'#a9adb8', borderDash:[5,5], pointRadius:0, borderWidth:1.5, tension:0},
          {label:'Selected capacity', data:[], borderColor:'#e0a93e', backgroundColor:'#e0a93e', pointRadius:6, showLine:false}
        ]},
        options:{responsive:true, maintainAspectRatio:false, animation:false,
          interaction:{mode:'nearest', intersect:false},
          plugins:{legend:{labels:{color:'#a9adb8', boxWidth:12}}, tooltip:{callbacks:{label:ctx => `${ctx.dataset.label}: ${ctx.parsed.y === null ? 'undefined' : ctx.parsed.y.toFixed(2) + '%'}`}}},
          scales:{x:{type:'linear', min:0, max:100, title:{display:true,text:'Investigation workload (% of validation)',color:'#a9adb8'}, ticks:{color:'#a9adb8'},grid:{color:'#30343f'}},
                  y:{min:0,max:100,title:{display:true,text:metric === 'recall' ? 'Historical fraud capture (%)' : 'Precision (%)',color:'#a9adb8'},ticks:{color:'#a9adb8'},grid:{color:'#30343f'}}},
          onClick:(event, elements) => {if (elements.length) {
            const x = chart.data.datasets[elements[0].datasetIndex].data[elements[0].index].x;
            select(points.reduce((best,p,index) => Math.abs(p.capacity_fraction*100-x)<Math.abs(points[best].capacity_fraction*100-x) ? index : best,0));
          }}
        }
      }); charts.push(chart);
    });
  }

  async function show() {
    if (aggregate) { for (const chart of charts) chart.resize(); return; }
    if (loading) return;
    loading = true; el('capacity-retry').hidden = true;
    el('capacity-status').textContent = 'Loading historical validation evidence…';
    const local = ['localhost','127.0.0.1'].includes(location.hostname) && location.port === '8080';
    const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 8000);
    try {
      const response = await fetch((local ? 'http://127.0.0.1:8000' : location.origin) + '/v1/dashboard/capacity', {signal:controller.signal});
      if (!response.ok) throw new Error('Evidence unavailable');
      const body = await response.json();
      if (body.schema_version !== '1.0.0' || body.population_id !== 'stage4-validation' || !Array.isArray(body.points) || body.points.length !== body.checkpoint_count || body.points[0].k !== 0 || body.points.at(-1).k !== body.n) throw new Error('Invalid evidence version');
      aggregate = body;
      el('capacity-workspace').hidden = false;
      el('capacity-slider').max = body.points.length - 1;
      el('capacity-count').max = body.n;
      el('capacity-population').textContent = `Validation: ${fmt(body.n)} eligible · ${fmt(body.positives)} verified fraud labels · prevalence ${pct(body.prevalence)} · AP ${body.average_precision.toFixed(5)} · aggregate v${body.schema_version}`;
      el('capacity-status').textContent = '';
      buildCharts(); select(2);
    } catch (error) {
      aggregate = null; el('capacity-workspace').hidden = true;
      el('capacity-status').textContent = 'Historical evidence could not be loaded. No simulator results are shown. Retry when the service is available.';
      el('capacity-retry').hidden = false;
    } finally {clearTimeout(timeout); loading = false;}
  }
  el('capacity-slider').addEventListener('input', event => {if(aggregate) select(Number(event.target.value));});
  el('capacity-count').addEventListener('change', event => {
    if (!aggregate) return;
    const n = event.target.valueAsNumber;
    if (!Number.isFinite(n)) {el('capacity-count').value = aggregate.points[selected].k; return;}
    const index = aggregate.points.reduce((best,p,index) => Math.abs(p.k - n)<Math.abs(aggregate.points[best].k - n) ? index : best, 0);
    select(index);
  });
  el('capacity-reset').addEventListener('click', () => {if(aggregate) select(2);});
  el('capacity-retry').addEventListener('click', show);
  window.CapacitySimulator = {show};
})();
