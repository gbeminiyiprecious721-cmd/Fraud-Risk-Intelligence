/* Browser-only controller. Case notes and assignments never leave the device. */
(() => {
  'use strict';
  const el=id=>document.getElementById(id);
  let store=null, selected=null, repaint=()=>{}, feedback='',batchRows=[];
  const options=(id,values)=>el(id).replaceChildren(...values.map(value=>{const o=document.createElement('option');o.value=value;o.textContent=value;return o;}));
  options('case-analyst',CaseStore.analysts);options('case-priority',CaseStore.priorities);options('case-status',CaseStore.statuses);
  function get(id){return store?.get(id)||null;}
  function render() {
    const c=selected?get(selected):null;
    el('case-open').hidden=!!c;el('case-open').disabled=!store;
    el('case-detail').hidden=!c;
    el('case-message').textContent=[store?.warning(),feedback].filter(Boolean).join(' ');
    if(!c)return;
    el('case-identity').textContent=`${c.id} · ${c.status} · created ${new Date(c.createdAt).toLocaleString()} · updated ${new Date(c.updatedAt).toLocaleString()}`;
    el('case-analyst').value=c.analyst;el('case-priority').value=c.priority;el('case-status').value=c.status;
    el('case-analyst').disabled=c.status==='Resolved';el('case-priority').disabled=c.status==='Resolved';
    el('case-add-note').disabled=c.status==='Resolved';
    el('case-notes').replaceChildren(...c.notes.map(n=>{const li=document.createElement('li');li.textContent=`${new Date(n.at).toLocaleString()} · ${n.text}`;return li;}));
    if(!c.notes.length){const li=document.createElement('li');li.textContent='No demo notes yet.';el('case-notes').append(li);}
    el('case-history').replaceChildren(...c.history.map(e=>{const li=document.createElement('li');li.textContent=`${new Date(e.at).toLocaleString()} · ${e.type}${e.before?' · '+e.before:''}${e.after?' → '+e.after:''}${e.text?' · '+e.text:''}`;return li;}));
  }
  function run(fn) {
    const fields=['case-analyst','case-priority','case-status','case-reason','case-note'];
    const draft=fields.map(id=>el(id).value);
    const openingWithFocus=document.activeElement===el('case-open');
    try {fn();feedback='Case updated locally.';repaint();render();if(openingWithFocus&&!el('case-detail').hidden)el('case-analyst').focus();}
    catch(error){feedback=error.message;render();fields.forEach((id,i)=>el(id).value=draft[i]);}
  }
  function filter(t) {
    const c=get(t.transaction_id),status=el('case-status-filter').value,analyst=el('case-analyst-filter').value,priority=el('case-priority-filter').value;
    return (!status||(status==='Not opened'?!c:c?.status===status))&&(!analyst||c?.analyst===analyst)&&(!priority||c?.priority===priority);
  }
  function emptyMessage() {
    const total=batchRows.length,cases=store?.list()||[];
    const status=el('case-status-filter').value,analyst=el('case-analyst-filter').value,priority=el('case-priority-filter').value;
    if(status==='Not opened'&&(analyst||priority))return `Not opened cannot match an analyst or priority: these belong to opened cases only. ${total} synthetic transactions remain in this batch. Clear case filters; saved cases are unchanged.`;
    if(status||analyst||priority) {
      if(!cases.length&&status!=='Not opened')return `No opened investigation cases in this batch. ${total} synthetic transactions are available. Clear case filters, then inspect a transaction to open a case. Seeded lane labels are not analyst decisions.`;
      const matches=batchRows.filter(filter).length;
      if(!matches)return status==='Not opened'
        ? `Every transaction in this ${total}-transaction batch already has an opened case. Clear case filters to view them.`
        : `No opened cases match the selected case filters. ${cases.length} opened cases and ${total} synthetic transactions exist in this batch. Clear case filters or inspect and update an existing case.`;
      return `Transaction filters exclude all ${matches} transactions matching the case selection in this ${total}-transaction batch. Clear all filters to broaden the view; notes, assignments and history are unchanged.`;
    }
    return `No synthetic transactions match the transaction filters. ${total} transactions remain available in this batch. Clear all filters to broaden the view; no cases were deleted.`;
  }
  function workload(rows) {
    const visible=new Set(rows.map(t=>t.transaction_id)),cases=(store?.list()||[]).filter(c=>visible.has(c.transactionId));
    const values=[['Opened cases',cases.length],['Active',cases.filter(c=>c.status!=='Resolved').length],['Unassigned active',cases.filter(c=>c.status!=='Resolved'&&c.analyst==='Unassigned').length],['Resolved',cases.filter(c=>c.status==='Resolved').length]];
    el('case-workload').replaceChildren(...values.map(([label,value])=>{const p=document.createElement('p');p.textContent=`${label}: ${value}`;return p;}));
    el('case-filter-help').textContent=`Case filters use opened cases only: ${store?.list().length||0} opened / ${batchRows.length} transactions. Seeded lane labels are not analyst decisions.`;
    el('case-storage-status').textContent=store?.warning()||'Browser-local synthetic cases · no server persistence · no real customer data';
  }
  el('case-open').addEventListener('click',()=>run(()=>store.open(selected)));
  el('case-save').addEventListener('click',()=>run(()=>{
    let c=get(selected);const target=el('case-status').value,reason=el('case-reason').value;
    const analyst=el('case-analyst').value,priority=el('case-priority').value;
    // Validate resolution requirements before applying any assignment change.
    if(target!==c.status&&(target==='Resolved'||c.status==='Resolved')&&(!reason.trim()||reason.length>300))throw Error('An explicit resolution or reopening reason is required.');
    if(c.status==='Resolved'&&target!=='Resolved'&&target!=='In Review')throw Error('Reopen to In Review with a reason.');
    const changes=[];
    if(c.status==='Resolved'&&target==='In Review')changes.push({field:'status',value:target,reason});
    if(c.analyst!==analyst)changes.push({field:'analyst',value:analyst});
    if(c.priority!==priority)changes.push({field:'priority',value:priority});
    if(c.status!=='Resolved'&&c.status!==target)changes.push({field:'status',value:target,reason});
    if(changes.length)store.update(selected,changes);
    el('case-reason').value='';
  }));
  el('case-add-note').addEventListener('click',()=>run(()=>{store.change(selected,'note',el('case-note').value);el('case-note').value='';}));
  el('case-reset').addEventListener('click',()=>{
    if(store&&window.confirm('Discard all locally saved demo case changes for the current seed and batch size? Notes, analyst assignments, priorities and case history will be deleted. Other demo batches are unchanged.'))run(()=>store.reset());
  });
  for(const id of ['case-status-filter','case-analyst-filter','case-priority-filter'])el(id).addEventListener('change',()=>repaint());
  el('case-clear-filters').addEventListener('click',()=>{for(const id of ['case-status-filter','case-analyst-filter','case-priority-filter'])el(id).value='';repaint();});
  window.CaseUI={
    setBatch(batch,refresh){batchRows=batch.transactions;repaint=refresh;selected=null;feedback='';el('case-reason').value='';el('case-note').value='';try{store=CaseStore.create(batch,{getItem:k=>localStorage.getItem(k),setItem:(k,v)=>localStorage.setItem(k,v),removeItem:k=>localStorage.removeItem(k)});}catch(error){store=null;feedback='Case management unavailable: '+error.message;}render();},
    show(id){if(selected!==id){el('case-note').value='';el('case-reason').value='';}selected=id;feedback='';render();},
    close(){selected=null;el('case-note').value='';el('case-reason').value='';},
    get,filter,emptyMessage,workload,status:t=>get(t.transaction_id)?.status||t.status,
    label:t=>{const c=get(t.transaction_id);return c?`${c.status} · ${c.analyst} · ${c.priority}`:'Not opened · seeded '+t.status;}
  };
})();
