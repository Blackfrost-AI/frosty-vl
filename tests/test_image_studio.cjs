const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {JSDOM} = require('jsdom');
const root = path.join(__dirname, '..');
const settle = () => new Promise(resolve => setImmediate(resolve));

async function studio() {
  const dom = new JSDOM(fs.readFileSync(path.join(root,'ui/image_studio.html'),'utf8'),
    {url:'http://localhost/image', runScripts:'outside-only'});
  const w = dom.window, requests=[];
  w.setInterval = () => 0;
  w.HTMLElement.prototype.scrollIntoView = () => {};
  w.HTMLCanvasElement.prototype.getContext = () => ({clearRect(){}});
  const photos=[{id:'image_'+'a'.repeat(32),name:'fixture.png',prompt:'A blue square',created_at:1,
    file_url:'/api/images/files/fixture.png',content_type:'image/png',width:256,height:256,dwm_scale:0.5}];
  let trashed=false;
  w.fetch = async (url, options={}) => {
    const payload=options.body?JSON.parse(options.body):null;
    requests.push({url,payload});
    let data={};
    if(url==='/api/images/health')data={ready:true,dwm:{enabled:true},dwm_default_scale:0.75,prompt_enhancement:{t2i:true,edit:true}};
    else if(url==='/api/workspaces')data={image:true,video:true};
    else if(url==='/api/images/gallery')data={ok:true,count:trashed?0:1,items:trashed?[]:photos,trash_count:trashed?1:0};
    else if(url==='/api/images/gallery/trash'&&payload){trashed=true;data={ok:true,results:[{ok:true,name:'fixture.png',trash_id:'b'.repeat(32)}]};}
    else if(url==='/api/images/gallery/trash')data={ok:true,items:trashed?[{id:'b'.repeat(32),name:'fixture.png',deleted_at:1,state:'trashed'}]:[]};
    else if(url==='/api/images/gallery/restore'){trashed=false;data={ok:true,results:[{ok:true,name:'fixture.png'}]};}
    else if(url==='/api/images/jobs')data={id:'img_'+'c'.repeat(24)};
    else if(url.startsWith('/api/images/jobs/'))data={status:'done',stage:'Complete',created_at:Date.now()/1000,seconds:1,outputs:[]};
    return {ok:true,json:async()=>data,blob:async()=>new w.Blob(['fixture'],{type:'image/png'})};
  };
  new vm.Script(fs.readFileSync(path.join(root,'ui/image_studio.js'),'utf8')).runInContext(dom.getInternalVMContext());
  // File decoding is exercised with real PNGs by Python and native browser checks.
  // These DOM tests focus on user actions, ordering and the actual submitted payload.
  w.eval('readReference=async file=>({name:file.name,data:"data:"+file.name,width:256,height:256})');
  await settle();
  return {w,requests,close:()=>dom.window.close()};
}

test('fresh form accepts ten references and submits them in reordered sequence', async()=>{
  const {w,requests,close}=await studio();
  try {
    assert.equal(w.document.querySelector('#references-section').hidden,false);
    assert.equal(w.document.querySelector('#video-workspace').hidden,false);
    assert.equal(w.document.querySelector('#dwm-scale').value,'0.75');
    await w.eval('addFiles(Array.from({length:10},(_,i)=>({name:"ref"+i+".png"})))');
    assert.equal(w.document.querySelector('#reference-count').textContent,'10 / 10');
    w.document.querySelector('[aria-label="Move reference right 1"]').click();
    w.document.querySelector('#prompt').value='Arrange these objects on a table';
    w.document.querySelector('#auto-enhance').checked=false;
    await w.eval('generate()');
    const request=requests.find(r=>r.url==='/api/images/jobs').payload;
    assert.equal(request.mode,'auto');
    assert.equal(request.images_b64.length,10);
    assert.deepEqual(request.images_b64.slice(0,2),['data:ref1.png','data:ref0.png']);
    assert.equal(request.dwm_scale,0.75);
    await w.eval('addFiles([{name:"overflow.png"}])');
    assert.match(w.document.querySelector('#form-error').textContent,/up to 10/);
    assert.equal(w.eval('state.refs.length'),10);
    w.document.querySelector('#clear-references').click();
    assert.equal(w.eval('state.refs.length'),0);
    assert.equal(w.document.querySelector('#generate span').textContent,'Generate image');
  } finally {close();}
});

test('concurrent additions remain bounded and gallery reuse appends',async()=>{
  const {w,close}=await studio();
  try {
    await w.eval('addFiles([{name:"original.png"}])');
    await w.eval('useAsReference("fixture.png")');
    assert.deepEqual(Array.from(w.eval('state.refs.map(r=>r.name)')),['original.png','fixture.png']);
    await w.eval('Promise.all([addFiles(Array.from({length:5},(_,i)=>({name:"a"+i}))),addFiles(Array.from({length:5},(_,i)=>({name:"b"+i})))])');
    assert.equal(w.eval('state.refs.length'),7);
    w.eval('setMode("masked")');
    assert.equal(w.document.querySelector('#reference-count').textContent,'7 / 9');
    assert.match(w.document.querySelector('#reference-hint').textContent,/9 references/);
  } finally {close();}
});

test('gallery delete, Undo, Trash view and restore use backend IDs',async()=>{
  const {w,requests,close}=await studio();
  try {
    await w.eval('trashImages([galleryItems[0].id])');
    assert.equal(w.document.querySelector('#gallery-count').textContent,'0');
    assert.equal(w.document.querySelector('#undo-trash').hidden,false);
    await w.document.querySelector('#undo-trash').onclick();
    assert.equal(w.document.querySelector('#gallery-count').textContent,'1');
    const deletion=requests.find(r=>r.url==='/api/images/gallery/trash'&&r.payload);
    assert.deepEqual(deletion.payload.ids,['image_'+'a'.repeat(32)]);
    assert.equal(requests.filter(r=>r.url==='/api/images/gallery/restore').length,1);
    await w.eval('trashImages([galleryItems[0].id])');
    w.document.querySelector('#show-trash').click();await settle();
    assert.equal(w.document.querySelector('#restore-selected').hidden,false);
    assert.match(w.document.querySelector('#gallery').textContent,/Restore/);
  } finally {close();}
});
