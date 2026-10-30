/* Versioned browser-local synthetic cases. No network calls or HTML rendering. */
(() => {
  'use strict';
  const seeds=[101,202,303,404,505], statuses=['New','In Review','Escalated','Resolved'];
  const analysts=['Unassigned','Fictional Analyst A','Fictional Analyst B','Fictional Analyst C'];
  const priorities=['Normal','High','Urgent'], bands=['Low','Medium','High','Critical'];
  const clone=value=>JSON.parse(JSON.stringify(value));
  const require=(condition,message)=>{if(!condition)throw Error(message);};
  const keys=(value,names)=>value && typeof value==='object' && !Array.isArray(value) && Object.keys(value).sort().join('|')===names.slice().sort().join('|');
  const text=(value,max)=>typeof value==='string'&&value.trim().length>0&&value.length<=max&&!/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/.test(value);
  const stamp=value=>typeof value==='string'&&Number.isFinite(Date.parse(value))&&new Date(value).toISOString()===value;
  const caseKeys=['schemaVersion','synthetic','id','transactionId','seed','batchSize','riskScore','riskBand','status','analyst','priority','notes','resolutionReason','createdAt','updatedAt','history'];
  function create(batch,storage) {
    require(batch?.synthetic===true&&seeds.includes(batch.seed)&&[24,48].includes(batch.transactions?.length),'Unsupported synthetic batch');
    const seed=batch.seed,size=batch.transactions.length,scope=`fri.demo.cases.v1.${seed}.${size}`,eligible=new Map();
    batch.transactions.forEach(t=>{
      require(Number.isInteger(t.sequence)&&t.sequence>=0&&t.sequence<size&&t.transaction_id===`TXN-DEMO-${String(seed).padStart(4,'0')}-${String(t.sequence).padStart(4,'0')}`&&!eligible.has(t.transaction_id)&&Number.isFinite(t.risk_score)&&t.risk_score>=0&&t.risk_score<=1&&bands.includes(t.risk_band),'Invalid synthetic transaction');
      eligible.set(t.transaction_id,clone(t));
    });
    let data={schemaVersion:1,seed,batchSize:size,revision:0,cases:[]},baseline=null,persistent=true,blocked=false,message='';
    const warn=reason=>{message=reason;};
    function base(t,at) {return {schemaVersion:1,synthetic:true,id:t.transaction_id.replace('TXN-','CASE-'),transactionId:t.transaction_id,seed,batchSize:size,riskScore:t.risk_score,riskBand:t.risk_band,status:'New',analyst:'Unassigned',priority:'Normal',notes:[],resolutionReason:'',createdAt:at,updatedAt:at,history:[]};}
    function apply(c,e) {
      require(keys(e,['type','at','before','after','text'])&&stamp(e.at)&&e.at>=c.updatedAt&&typeof e.before==='string'&&typeof e.after==='string'&&typeof e.text==='string','Invalid case event');
      require(c.history.length<64,'Activity limit reached (64). Reset this demo batch to start again.');
      if(e.type==='created') require(!c.history.length&&e.before===''&&e.after==='New'&&e.text===''&&e.at===c.createdAt,'Invalid creation');
      else {
        require(c.history.length>0,'Creation event required');
        if(['status','resolved','reopened'].includes(e.type)) {
          require(statuses.includes(e.after)&&e.before===c.status&&e.after!==c.status,'Invalid status transition');
          if(c.status==='Resolved')require(e.type==='reopened'&&e.after==='In Review'&&text(e.text,300),'Reopen to In Review with a reason (1–300 characters).');
          else if(e.after==='Resolved')require(e.type==='resolved'&&text(e.text,300),'Resolution requires a reason (1–300 characters).');
          else require(e.type==='status'&&e.text==='','Invalid status event');
          c.status=e.after;c.resolutionReason=e.type==='resolved'?e.text:'';
        } else if(e.type==='analyst'||e.type==='priority') {
          require(c.status!=='Resolved','Reopen the case before editing.');
          require((e.type==='analyst'?analysts:priorities).includes(e.after)&&e.before===c[e.type]&&e.after!==e.before&&e.text==='','Invalid assignment or priority');c[e.type]=e.after;
        } else if(e.type==='note') {
          require(c.status!=='Resolved','Reopen the case before adding notes.');
          require(text(e.text,500)&&e.before===''&&e.after===''&&c.notes.length<16,'Notes require 1–500 characters; maximum 16 per case.');c.notes.push({at:e.at,text:e.text});
        } else throw Error('Unsupported event');
      }
      c.history.push(clone(e));c.updatedAt=e.at;
    }
    function validate(value) {
      require(keys(value,['schemaVersion','seed','batchSize','revision','cases'])&&value.schemaVersion===1&&value.seed===seed&&value.batchSize===size&&Number.isInteger(value.revision)&&value.revision>=0&&value.revision<=100000&&Array.isArray(value.cases)&&value.cases.length<=size,'Unsupported or malformed stored schema');
      const seen=new Set();
      for(const c of value.cases) {
        const t=eligible.get(c.transactionId);
        require(keys(c,caseKeys)&&t&&!seen.has(c.transactionId)&&stamp(c.createdAt)&&Array.isArray(c.history)&&c.history.length>=1&&c.history.length<=64,'Invalid stored case');seen.add(c.transactionId);
        const replay=base(t,c.createdAt);c.history.forEach(e=>apply(replay,e));
        for(const key of caseKeys)require(JSON.stringify(c[key])===JSON.stringify(replay[key]),'Stored case does not match validated activity');
      }
      return value;
    }
    try {
      baseline=storage.getItem(scope);
      if(baseline!==null) {require(typeof baseline==='string'&&baseline.length<=393216,'Stored demo exceeds size limit');data=validate(JSON.parse(baseline));}
    } catch(error) {
      if(baseline!==null){blocked=true;warn('Stored demo cases are invalid or unsupported. Reset Demo Cases to recover; nothing was overwritten.');}
      else {persistent=false;warn('Browser storage unavailable. Changes remain in memory only and will be lost on refresh.');}
    }
    function save(next) {
      require(!blocked,'Reset Demo Cases to recover invalid storage.');
      validate(next);const encoded=JSON.stringify(next);require(encoded.length<=393216,'Local demo storage limit reached. Reset this batch to start again.');
      if(persistent) {
        let current;
        try{current=storage.getItem(scope);}catch(error){persistent=false;warn('Storage unavailable. Changes are memory-only; refresh will lose them.');}
        if(persistent&&current!==baseline){blocked=true;warn('Another tab changed this demo. Reload before editing, or explicitly reset.');throw Error(message);}
        if(persistent)try{storage.setItem(scope,encoded);baseline=encoded;}catch(error){persistent=false;warn('Could not save (quota or storage denial). Changes remain in memory only; refresh will lose them.');}
      }
      data=next;
    }
    function current(id){require(eligible.has(id),'Transaction is not in the current synthetic batch');return data.cases.find(c=>c.transactionId===id);}
    function eventTime(c){return new Date(Math.max(Date.now(),Date.parse(c?.updatedAt||0)||0)).toISOString();}
    function update(id,changes) {
      require(Array.isArray(changes)&&changes.length>=1&&changes.length<=3,'Invalid edit');
      const next=clone(data),c=next.cases.find(c=>c.transactionId===id);require(c&&eligible.has(id),'Open a case first');
      for(const {field,value,reason=''} of changes) {
        require(typeof value==='string'&&typeof reason==='string','Invalid change');
        let type=field,before=field==='note'?'':c[field],after=field==='note'?'':value,detail=field==='note'?value:reason;
        if(field==='status'){type=c.status==='Resolved'?'reopened':value==='Resolved'?'resolved':'status';if(type==='status')detail='';}
        require(['status','analyst','priority','note'].includes(field),'Unsupported change');
        apply(c,{type,at:eventTime(c),before,after,text:detail});

      }
      next.revision++;save(next);return clone(c);
    }
    return {
      get:id=>{const c=current(id);return c?clone(c):null;},list:()=>clone(data.cases),warning:()=>message,
      open(id) {
        const existing=current(id);if(existing)return clone(existing);
        const next=clone(data),c=base(eligible.get(id),eventTime());apply(c,{type:'created',at:c.createdAt,before:'',after:'New',text:''});next.cases.push(c);next.revision++;save(next);return clone(c);
      },
      change:(id,field,value,reason='')=>update(id,[{field,value,reason}]),
      update,
      reset() {
        data={schemaVersion:1,seed,batchSize:size,revision:0,cases:[]};blocked=false;message='';baseline=null;persistent=true;
        try{storage.removeItem(scope);}catch(error){persistent=false;warn('Reset in memory only: browser storage could not be cleared.');}
      }
    };
  }
  window.CaseStore=Object.freeze({create,statuses:Object.freeze(statuses),analysts:Object.freeze(analysts),priorities:Object.freeze(priorities)});
})();
