/* Fixed aggregate/global and allowlisted synthetic explanations; no case writes. */
(() => {
 'use strict';
 const el=id=>document.getElementById(id),fmt=v=>Number(v).toFixed(6),signed=v=>(v>0?'+':'')+fmt(v);
 const api=()=>['localhost','127.0.0.1'].includes(location.hostname)&&location.port==='8080'?'http://127.0.0.1:8000':location.origin;
 let data=null,batch=null,loading=false,globalChart=null,localChart=null,requestToken=0,pending=null,batchKey='';
 const chartOptions={indexAxis:'y',responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:'#a9adb8',font:{size:10}},grid:{color:'#30343f'}},y:{ticks:{color:'#a9adb8',autoSkip:false,font:{size:10}},grid:{color:'#30343f'}}}};
 function row(values){const r=document.createElement('tr');for(const value of values){const c=document.createElement('td');c.textContent=value;r.append(c);}return r;}
 function chart(id,labels,values,colors){if(typeof Chart==='undefined')return null;return new Chart(el(id),{type:'bar',data:{labels,datasets:[{label:'Model explanation',data:values,backgroundColor:colors}]},options:chartOptions});}
 async function fetchJSON(path){const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),8000);try{const r=await fetch(api()+path,{signal:controller.signal});if(!r.ok)throw new Error('Unavailable');return await r.json();}finally{clearTimeout(timer);}}
 function featureDetail(){
  if(!data)return;const r=data.methods[el('explanation-method').value]?.find(r=>r.name===el('explanation-feature').value);if(!r)return;
  const dictionary={TransactionAmt:'Transaction amount input; historical currency is not established.',ProductCD:'Coded product category; codes do not establish business causes.',card4:'Card-network category.',card6:'Coded card type.',dist1:'Dataset distance input; units are not established.'};
  const c=r.components,definition=dictionary[r.name]||'Anonymised/coded input; precise business meaning is not verified.';
  el('explanation-feature-detail').textContent=`${r.name}: ${definition} Group includes ${c.numeric} numeric, ${c.missingness_indicator} missingness and ${c.categorical_onehot} one-hot components. Importance ${fmt(r.value)} under the selected method is model reliance, not a causal effect or fraud verdict.`;
 }
 function renderGlobal(){
  if(!data)return;let method=el('explanation-method').value;if(!data.methods[method])method='mdi';el('explanation-method').value=method;
  el('explanation-method').querySelector('option[value="shap"]').disabled=!data.capabilities.global_shap;
  const rows=data.methods[method],previous=el('explanation-feature').value;
  el('explanation-feature').replaceChildren(...rows.map(r=>{const option=document.createElement('option');option.value=r.name;option.textContent=r.name;return option;}));
  if(rows.some(r=>r.name===previous))el('explanation-feature').value=previous;
  el('explanation-provenance').textContent=`Frozen RF · 813 ordered columns / 392 raw groups · ${data.capabilities.global_shap?`TreeSHAP: ${data.population.n} label-free validation examples, not a pristine holdout`:`SHAP unavailable; MDI from fitted trees only`} · model ${data.model_sha256.slice(0,12)}…`;
  el('explanation-global-title').textContent=method==='mdi'?'Impurity importance · top10':'Mean absolute grouped TreeSHAP · top10';
  el('explanation-global-note').textContent=method==='mdi'?'Fitted impurity-reduction share; biased by cardinality/correlation. Not permutation sensitivity.':'Signed components summed per raw input before averaging absolute values. Score units, not calibrated probabilities.';
  globalChart?.destroy();globalChart=chart('explanation-global-chart',rows.slice(0,10).map(r=>r.name),rows.slice(0,10).map(r=>r.value),method==='mdi'?'#5aa9ff':'#6ed8b5');
  el('explanation-global-table').replaceChildren(...rows.map(r=>row([r.name,fmt(r.value)])));
  const s=data.stability;
  el('explanation-stability').textContent=s.status==='available'?`200 row-weighted temporal-block resamples (16 blocks). Top10 overlap min/median/max: ${s.overlap_min.toFixed(2)} / ${s.overlap_median.toFixed(2)} / ${s.overlap_max.toFixed(2)}. Rank variation is descriptive sample sensitivity, not a confidence interval or robustness guarantee.`:'Stability evidence unavailable: '+s.reason;
  featureDetail();
 }
 function clearLocal(message){localChart?.destroy();localChart=null;el('explanation-local-status').textContent=message;el('explanation-local-table').replaceChildren();el('explanation-reconciliation').textContent='';}
 async function loadLocal(){
  const id=el('explanation-transaction').value,token=++requestToken;
  el('explanation-local-retry').hidden=true;
  if(!data?.capabilities.local_shap){clearLocal('Validated local explanations are unavailable; use the validated global methods.');return;}
  if(!id){clearLocal('No supported synthetic transaction selected. Open a demo transaction in the Investigation Queue.');return;}
  clearLocal('Loading fixed synthetic explanation…');
  try{
   const record=await fetchJSON('/v1/dashboard/explanation?transaction_id='+encodeURIComponent(id));if(token!==requestToken)return;
   const expected=batch?.transactions.find(t=>t.transaction_id===id);
   if(record.transaction_id!==id||!expected||!Number.isFinite(record.baseline)||!Number.isFinite(record.score)||record.score<0||record.score>1||record.contributions?.length!==10||!Number.isFinite(record.other)||!Number.isFinite(record.other_positive)||!Number.isFinite(record.other_negative)||Math.abs(record.other_positive+record.other_negative-record.other)>1e-8||!record.contributions.every(c=>typeof c.name==='string'&&Number.isFinite(c.value)))throw new Error('Invalid local data');
   const total=record.baseline+record.contributions.reduce((sum,c)=>sum+c.value,0)+record.other;
   if(Math.abs(total-record.score)>1e-8||Math.abs(record.score-expected.risk_score)>5.00001e-7)throw new Error('Score mismatch');
   const values=[...record.contributions,{name:'Other increases',value:record.other_positive},{name:'Other decreases',value:record.other_negative}];
   localChart=chart('explanation-local-chart',values.map(c=>c.name),values.map(c=>c.value),values.map(c=>c.value>0?'#e0a47a':'#6ed8b5'));
   el('explanation-local-status').textContent=`${id} · + raises / − lowers score relative to baseline; not causal.`;
   el('explanation-reconciliation').textContent=`Synthetic score reconciliation: baseline ${fmt(record.baseline)} + signed groups + Other = score ${fmt(record.score)} (residual ${Math.abs(total-record.score).toExponential(1)}). Descriptive case attributes remain separate; no labels or notes are used.`;
   el('explanation-local-table').replaceChildren(row(['Baseline',fmt(record.baseline)]),...record.contributions.map(c=>row([c.name,signed(c.value)])),row(['Other features (net)',signed(record.other)]),row(['↳ Other increases (included)',signed(record.other_positive)]),row(['↳ Other decreases (included)',signed(record.other_negative)]),row(['Actual frozen score',fmt(record.score)]));
  }catch(error){if(token!==requestToken)return;clearLocal('Local explanation unavailable for this identifier, or assets/score do not match. Only fixed supported synthetic cohorts can be explained. Global explanations remain usable.');el('explanation-local-retry').hidden=false;}
 }
 function setBatch(value){
  batch=value;const key=(value?.transactions||[]).map(t=>t.transaction_id).join('|')+'|'+value?.model?.artifact_sha256;
  if(key===batchKey)return;batchKey=key;const previous=pending||el('explanation-transaction').value;
  el('explanation-transaction').replaceChildren(...(value?.transactions||[]).map(t=>{const o=document.createElement('option');o.value=t.transaction_id;o.textContent=t.transaction_id;return o;}));
  if((value?.transactions||[]).some(t=>t.transaction_id===previous))el('explanation-transaction').value=previous;
  if(data&&data.model_sha256!==value?.model?.artifact_sha256){data=null;globalChart?.destroy();globalChart=null;clearLocal('');el('explanation-workspace').hidden=true;}
  if(document.body.classList.contains('explanation-active')){if(data)loadLocal();else show();}
 }
 async function show(){
  if(!batch?.model?.artifact_sha256){el('explanation-status').textContent='Waiting for the demo batch identity. Reload the dashboard if the service is unavailable.';return;}
  if(data){if(pending){el('explanation-transaction').value=pending;pending=null;loadLocal();}globalChart?.resize();localChart?.resize();return;}
  if(loading)return;loading=true;el('explanation-retry').hidden=true;el('explanation-status').textContent='Loading verified frozen-model explanations…';
  try{
   const body=await fetchJSON('/v1/dashboard/explainability');
   if(body.schema_version!=='1.0.0'||body.model_sha256!==batch?.model?.artifact_sha256||!Array.isArray(body.methods?.mdi)||body.methods.mdi.length!==20||!body.capabilities?.mdi||body.population?.n!==512)throw new Error('Invalid identity/resource');
   data=body;el('explanation-workspace').hidden=false;el('explanation-status').textContent=typeof Chart==='undefined'?'Charts unavailable; verified numerical tables remain usable.':body.unavailable_reason||'';
   el('explanation-limitations').replaceChildren(...body.limitations.map(text=>{const p=document.createElement('p');p.textContent=text;return p;}));
   renderGlobal();if(pending){el('explanation-transaction').value=pending;pending=null;}await loadLocal();
  }catch(error){data=null;globalChart?.destroy();globalChart=null;clearLocal('');el('explanation-workspace').hidden=true;el('explanation-status').textContent='Validated explanation resources unavailable or mismatched. Existing dashboard and cases remain operational.';el('explanation-retry').hidden=false;}
  finally{loading=false;}
 }
 el('explanation-method').addEventListener('change',renderGlobal);el('explanation-feature').addEventListener('change',featureDetail);
 el('explanation-transaction').addEventListener('change',loadLocal);el('explanation-local-retry').addEventListener('click',loadLocal);
 el('explanation-reset').addEventListener('click',()=>{if(!data)return;el('explanation-method').value=data.capabilities.global_shap?'shap':'mdi';el('explanation-feature').value='';if(batch?.transactions.length)el('explanation-transaction').value=batch.transactions[0].transaction_id;renderGlobal();loadLocal();});
 el('explanation-retry').addEventListener('click',show);
 window.ModelExplanations={setBatch,show,selectTransaction(id){pending=id;el('explanation-transaction').value=id;}};
})();
