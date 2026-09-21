const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const {JSDOM}=require('jsdom');
const root=path.join(__dirname,'..');
const settle=()=>new Promise(resolve=>setImmediate(resolve));
async function studio(saved={},ready=true){
  const dom=new JSDOM(fs.readFileSync(path.join(root,'ui/video_studio.html'),'utf8'),{url:'http://localhost/video',runScripts:'outside-only'});
  const w=dom.window,requests=[],jobs=[];let trashed=false;
  w.setInterval=()=>0;w.HTMLElement.prototype.scrollIntoView=()=>{};
  w.HTMLDialogElement.prototype.showModal=function(){this.open=true;};w.HTMLDialogElement.prototype.close=function(){this.open=false;};
  for(const [k,v]of Object.entries(saved))w.localStorage.setItem(k,v);
  const engine={id:'video',label:'Video fixture',description:'No model required',capabilities:['text_to_video','image_to_video','scene_lab'],health:{ready},controls:{durations:[5,8],steps:{min:4,max:60,default:30},resolutions:['480p'],aspect_ratios:['16:9']}};
  const item={id:'video_'+'a'.repeat(32),name:'fixture.mp4',prompt:'A blue boat',seed:42,duration_seconds:5,num_inference_steps:24,engine_id:'video',engine_label:'Video fixture',file_url:'/api/videos/files/fixture.mp4'};
  w.fetch=async(url,options={})=>{
    const payload=options.body?JSON.parse(options.body):null;requests.push({url,payload});let data={};
    if(url==='/api/workspaces')data={image:true,video:true};
    else if(url==='/api/engines')data={ok:true,default:'image',engines:[{id:'image',capabilities:['text_to_image']},engine]};
    else if(url==='/api/videos/jobs'&&payload){const job={id:'vid_'+String(jobs.length+1).padStart(24,'0'),...payload,status:'queued',stage:'Queued'};jobs.unshift(job);data=job;}
    else if(url==='/api/videos/jobs')data={ok:true,items:jobs};
    else if(url.endsWith('/cancel')){const job=jobs.find(j=>url.includes(j.id));job.status='cancelled';job.stage='Cancelled before rendering';data=job;}
    else if(url==='/api/videos/gallery')data={ok:true,items:trashed?[]:[item]};
    else if(url==='/api/videos/gallery/trash'&&payload){trashed=true;data={ok:true,results:[{ok:true,trash_id:'b'.repeat(32)}]};}
    else if(url==='/api/videos/gallery/trash')data={ok:true,items:trashed?[{id:'b'.repeat(32),name:item.name}]:[]};
    else if(url==='/api/videos/gallery/restore'){trashed=false;data={ok:true,results:[{ok:true,name:item.name}]};}
    else if(url==='/api/images/gallery')data={items:[{name:'opening.png',prompt:'Opening frame',file_url:'/api/images/files/opening.png'}]};
    return {ok:true,json:async()=>data,blob:async()=>new w.Blob(['fixture'],{type:'image/png'})};
  };
  new vm.Script(fs.readFileSync(path.join(root,'ui/video_studio.js'),'utf8')).runInContext(dom.getInternalVMContext());
  w.eval('imageFile=async file=>"data:image/png;base64,fixture"');await settle();
  return {w,requests,jobs,engine,close:()=>dom.window.close()};
}
test('queue submits supported controls, returns immediately and survives page refresh',async()=>{
  const s=await studio();let saved;
  try{const {w,requests,jobs}=s;w.document.querySelector('#p').value='A boat at sunset';await w.eval('generate()');
    const sent=requests.find(r=>r.url==='/api/videos/jobs'&&r.payload).payload;
    assert.equal(sent.engine_id,'video');assert.equal(sent.num_inference_steps,30);assert.equal(sent.identity_lock,false);assert.ok(sent.request_id);
    assert.equal(w.document.querySelector('#go').disabled,false);assert.equal(w.document.querySelector('#jobs-count').textContent,'1');
    assert.equal(w.document.querySelector('#video-jobs button').textContent,'Cancel queued');
    jobs[0].status='done';jobs[0].stage='Complete';jobs[0].file_url='/api/videos/files/fixture.mp4';jobs[0].used_seed=42;await w.eval('refreshJobs()');
    assert.match(w.document.querySelector('#result video').src,/fixture.mp4/);saved=Object.fromEntries(['frosty.video.draft','frosty.video.jobs'].map(k=>[k,w.localStorage.getItem(k)]));
  }finally{s.close();}
  const restored=await studio(saved);try{assert.equal(restored.w.document.querySelector('#p').value,'A boat at sunset');assert.match(restored.w.document.querySelector('#status').textContent,/Check the gallery/);}finally{restored.close();}
});
test('queue cancellation uses the selected job ID and offline drafts do not submit',async()=>{
  const s=await studio();try{s.w.document.querySelector('#p').value='Boat';await s.w.eval('generate()');await s.w.document.querySelector('#video-jobs button').onclick();assert.equal(s.jobs[0].status,'cancelled');assert.equal(s.w.document.querySelector('#jobs-count').textContent,'0');s.engine.health.ready=false;await s.w.eval('refreshEngines();');await s.w.eval('generate()');assert.match(s.w.document.querySelector('#status').textContent,/offline/);assert.equal(s.requests.filter(x=>x.url==='/api/videos/jobs'&&x.payload).length,1);}finally{s.close();}
});
test('scene editing keeps focus during health refresh, supports order and submits scene data',async()=>{
  const s=await studio();try{const {w}=s;const field=w.document.querySelector('#scenes textarea');field.focus();field.value='First custom scene';field.dispatchEvent(new w.Event('input',{bubbles:true}));await w.eval('refreshEngines()');assert.equal(w.document.activeElement,field);
    w.document.querySelectorAll('.scene-actions button')[1].click();assert.equal(w.document.querySelectorAll('#scenes textarea')[1].value,'First custom scene');w.document.querySelectorAll('.scene-actions button')[2].click();assert.equal(w.document.querySelectorAll('.scene').length,4);
    await w.eval('generateLab()');const sent=s.requests.find(r=>r.url==='/api/videos/jobs'&&r.payload).payload;assert.equal(sent.kind,'scenes');assert.equal(sent.scenes[2].prompt,'First custom scene');assert.equal(sent.continuity,true);
  }finally{s.close();}
});
test('gallery supports selection, delete, Undo, Trash, restore and reusing settings',async()=>{
  const s=await studio();try{const {w}=s;await w.eval('libraryAction("trash",[galleryItems[0].id])');assert.equal(w.document.querySelectorAll('.tile').length,0);assert.equal(w.document.querySelector('#undo-video-trash').hidden,false);await w.document.querySelector('#undo-video-trash').onclick();assert.equal(w.document.querySelectorAll('.tile').length,1);
    w.document.querySelector('.tile .ghost').click();assert.equal(w.document.querySelector('#p').value,'A blue boat');assert.equal(w.document.querySelector('#s').value,'42');assert.equal(w.document.querySelector('#steps').value,'24');
    w.document.querySelector('#select-all-videos').checked=true;w.document.querySelector('#select-all-videos').onchange();await w.document.querySelector('#delete-videos').onclick();w.document.querySelector('#video-trash').click();await settle();assert.equal(w.document.querySelector('.tile button').textContent,'Restore');await w.document.querySelector('.tile button').onclick();assert.equal(w.document.querySelectorAll('.tile').length,0);
    assert.deepEqual(s.requests.find(r=>r.url==='/api/videos/gallery/trash'&&r.payload).payload.ids,['video_'+'a'.repeat(32)]);
  }finally{s.close();}
});
test('Frosty Image handoff and editable shot builder are included in the video request',async()=>{
  const s=await studio();try{const {w}=s;await w.document.querySelector('#choose-image-library').onclick();await w.document.querySelector('#image-library-items button').onclick();assert.equal(w.document.querySelector('#image-library-dialog').open,false);
    w.document.querySelector('#p').value='Opening scene';w.document.querySelector('#shot-camera').value='Slow dolly forward';w.document.querySelector('#apply-shot').click();w.document.querySelector('#quality-preset').value='draft';w.document.querySelector('#quality-preset').onchange();await w.eval('generate()');const payload=s.requests.find(x=>x.url==='/api/videos/jobs'&&x.payload).payload;assert.match(payload.prompt,/Camera: Slow dolly forward/);assert.equal(payload.num_inference_steps,18);assert.match(payload.image_b64,/data:image\/png/);
  }finally{s.close();}
});
