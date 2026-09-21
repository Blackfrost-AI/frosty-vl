'use strict';
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const state = {mode:'auto', refs:[], ready:false, enhancementReady:false, enhancement:null, job:null, tool:'brush', drawing:false, start:null, paintSnapshot:null};
const sizes = {'1:1':[2048,2048],'4:3':[2400,1792],'3:4':[1792,2400],'3:2':[2528,1696],'2:3':[1696,2528],'16:9':[2752,1536],'9:16':[1536,2752]};
const hints = {auto:'Create with a prompt, or add references to edit and compose.',generate:'Describe the image you want to create.',edit:'Describe how to change or combine your references.',transparent:'Create a transparent image or edit an existing RGBA layer.',extract:'Name the subject to keep on a transparent background.',masked:'Describe the change inside your painted region.',annotate:'Draw circles or marks, then describe your edit.'};
const paint = $('#paint'), ctx = paint.getContext('2d', {willReadFrequently:true});

function dimensions(){if($('#custom-size').checked)return [Number($('#custom-width').value),Number($('#custom-height').value)];const scale=Number($('#resolution').value)/2048;return sizes[$('#ratio').value].map(v=>Math.round(v*scale/32)*32);}
function updateDimensions(){const [w,h]=dimensions();$('#dimensions').textContent=`${w} × ${h}`;}
$('#ratio').onchange=updateDimensions;$('#resolution').onchange=updateDimensions;$('#custom-width').oninput=$('#custom-height').oninput=updateDimensions;$('#custom-size').onchange=()=>{$('#custom-dimensions').hidden=!$('#custom-size').checked;$('#ratio').disabled=$('#resolution').disabled=$('#custom-size').checked;if($('#custom-size').checked)$('#auto-ratio').checked=false;updateDimensions();};$('#auto-enhance').onchange=()=>{if(!$('#auto-enhance').checked)$('#auto-ratio').checked=false;};$('#auto-ratio').onchange=()=>{if($('#auto-ratio').checked){$('#auto-enhance').checked=true;$('#custom-size').checked=false;$('#custom-size').onchange();}};
function formError(message=''){$('#form-error').textContent=message;}
function setMode(mode){
  if(state.mode!==mode)ctx.clearRect(0,0,paint.width,paint.height);
  state.mode=mode;
  $$('[data-mode]').forEach(b=>{b.classList.toggle('active',b.dataset.mode===mode);b.setAttribute('aria-pressed',String(b.dataset.mode===mode));});
  $('#mode-hint').textContent=mode==='auto'&&state.refs.length?'Describe how to edit or combine your references.':hints[mode];
  $('#generate span').textContent=(mode==='auto'&&!state.refs.length)||mode==='transparent'?'Generate image':'Apply edit';
  $('#automatic-mode-label').textContent=state.refs.length?'Edit / compose · automatic':'Create / edit automatically';
  $('#reference-count').textContent=`${state.refs.length} / ${mode==='masked'?9:10}`;
  $('#reference-hint').textContent=mode==='masked'?'Up to 9 references + 1 mask. Image 1 is the canvas; white marks the edit region.':'No images: create something new. Add up to ten references to edit or compose. Mention “image 1” or “image 2” in your direction.';
  invalidateEnhancement();refreshEditor();formError();
}
$$('[data-mode]').forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
$$('[data-example]').forEach(b=>b.onclick=()=>{$('#prompt').value=b.dataset.example;$('#prompt').focus();});

function loadImage(url){return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(new Error('This image could not be read.'));image.src=url;});}
async function readReference(file){if(!['image/png','image/jpeg','image/webp'].includes(file.type))throw new Error('Choose PNG, JPEG, or WebP images.');if(file.size>20_000_000)throw new Error('Upload images smaller than 20 MB.');const url=URL.createObjectURL(file);try{const image=await loadImage(url);if(image.naturalWidth*image.naturalHeight>16_000_000)throw new Error('Reference images must be at most 16 megapixels.');const scale=Math.min(1,1600/Math.max(image.naturalWidth,image.naturalHeight));const canvas=document.createElement('canvas');canvas.width=Math.round(image.naturalWidth*scale);canvas.height=Math.round(image.naturalHeight*scale);canvas.getContext('2d').drawImage(image,0,0,canvas.width,canvas.height);return {name:file.name,data:canvas.toDataURL('image/png'),width:canvas.width,height:canvas.height};}finally{URL.revokeObjectURL(url);}}
function addFiles(files){
  return ingestReferences(async()=>{const added=[];for(const file of files)added.push(await readReference(file));return added;});
}
$('#references').onchange=async e=>{await addFiles([...e.target.files]);e.target.value='';};
const drop=$('#drop-zone');drop.ondragover=e=>{e.preventDefault();drop.classList.add('drag');};drop.ondragleave=()=>drop.classList.remove('drag');drop.ondrop=e=>{e.preventDefault();drop.classList.remove('drag');addFiles([...e.dataTransfer.files]);};
document.addEventListener('paste',e=>{const files=[...(e.clipboardData?.files||[])].filter(f=>f.type.startsWith('image/'));if(files.length){e.preventDefault();addFiles(files);}});
function renderRefs(){
  const root=$('#reference-list');root.replaceChildren();
  $('#reference-count').textContent=`${state.refs.length} / ${state.mode==='masked'?9:10}`;
  $('#clear-references').disabled=!state.refs.length;
  state.refs.forEach((ref,index)=>{
    const item=document.createElement('div');item.className='reference';item.draggable=true;
    item.ondragstart=e=>{e.dataTransfer.setData('application/x-frosty-reference',String(index));item.classList.add('active-drag');};
    item.ondragend=()=>item.classList.remove('active-drag');
    item.ondragover=e=>{e.preventDefault();item.classList.add('over');};item.ondragleave=()=>item.classList.remove('over');
    item.ondrop=e=>{e.preventDefault();item.classList.remove('over');const raw=e.dataTransfer.getData('application/x-frosty-reference');if(!/^\d+$/.test(raw))return;const from=Number(raw);if(from<state.refs.length){const [moved]=state.refs.splice(from,1);state.refs.splice(index,0,moved);referencesChanged();}};
    const image=document.createElement('img');image.src=ref.data;image.alt=`Reference ${index+1}: ${ref.name}`;
    image.tabIndex=0;image.onclick=()=>previewReference(ref,index);image.onkeydown=e=>{if(e.key==='Enter')previewReference(ref,index);};
    const title=document.createElement('div');title.className='ref-name';title.textContent=ref.name;title.title=`${ref.name} · ${ref.width} × ${ref.height}`;
    const tools=document.createElement('div');tools.className='ref-tools';const number=document.createElement('span');number.textContent=String(index+1);tools.append(number);
    const actions=[['←','Move reference left',()=>{if(index>0){[state.refs[index-1],state.refs[index]]=[state.refs[index],state.refs[index-1]];referencesChanged();}}],['→','Move reference right',()=>{if(index<state.refs.length-1){[state.refs[index+1],state.refs[index]]=[state.refs[index],state.refs[index+1]];referencesChanged();}}],['↻','Replace reference',()=>replaceReference(index)],['×','Remove reference',()=>{state.refs.splice(index,1);referencesChanged();}]];
    for(const [label,title,fn] of actions){const button=document.createElement('button');button.textContent=label;button.title=title;button.setAttribute('aria-label',`${title} ${index+1}`);button.onclick=fn;tools.append(button);}
    item.append(image,title,tools);root.append(item);
  });refreshEditor();
}
function refreshEditor(reset=false){const show=['masked','annotate'].includes(state.mode)&&state.refs.length>0;$('#editor').hidden=!show;if(!show)return;const ref=state.refs[0];$('#editor-image').src=ref.data;if(reset||paint.width!==ref.width||paint.height!==ref.height){paint.width=ref.width;paint.height=ref.height;}const mask=state.mode==='masked';$('#editor-title').textContent=mask?'Paint an edit region':'Draw your editing annotations';$('#preserve-label').hidden=!mask;$('#mask-upload-label').hidden=!mask;$('#editor-hint').textContent=mask?'Paint the region to change, or upload a mask: white changes, black stays.':'Draw a circle or mark on the reference. Explain what to change in your direction.';}
$('#clear-paint').onclick=()=>ctx.clearRect(0,0,paint.width,paint.height);
for(const tool of ['brush','circle'])$('#'+tool).onclick=()=>{state.tool=tool;$('#brush').classList.toggle('selected',tool==='brush');$('#circle').classList.toggle('selected',tool==='circle');};
function point(event){const box=paint.getBoundingClientRect();return {x:(event.clientX-box.left)*paint.width/box.width,y:(event.clientY-box.top)*paint.height/box.height};}
paint.onpointerdown=e=>{e.preventDefault();paint.setPointerCapture(e.pointerId);state.drawing=true;state.start=point(e);state.paintSnapshot=ctx.getImageData(0,0,paint.width,paint.height);ctx.strokeStyle=state.mode==='masked'?'rgba(255,85,100,1)':'#ff3344';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=Number($('#brush-size').value)*paint.width/Math.max(paint.getBoundingClientRect().width,1);ctx.lineCap='round';ctx.lineJoin='round';ctx.beginPath();ctx.moveTo(state.start.x,state.start.y);if(state.tool==='brush'){ctx.arc(state.start.x,state.start.y,ctx.lineWidth/2,0,2*Math.PI);ctx.fill();ctx.beginPath();ctx.moveTo(state.start.x,state.start.y);}};
paint.onpointermove=e=>{if(!state.drawing)return;const p=point(e);if(state.tool==='circle'){ctx.putImageData(state.paintSnapshot,0,0);ctx.beginPath();ctx.ellipse((p.x+state.start.x)/2,(p.y+state.start.y)/2,Math.abs(p.x-state.start.x)/2,Math.abs(p.y-state.start.y)/2,0,0,2*Math.PI);if(state.mode==='masked')ctx.fill();else ctx.stroke();}else{ctx.lineTo(p.x,p.y);ctx.stroke();}};
paint.onpointerup=paint.onpointercancel=()=>{state.drawing=false;state.paintSnapshot=null;};
function maskData(){const pixels=ctx.getImageData(0,0,paint.width,paint.height);let marked=false;for(let i=0;i<pixels.data.length;i+=4){const v=pixels.data[i+3];marked ||= v>0;pixels.data[i]=pixels.data[i+1]=pixels.data[i+2]=v;pixels.data[i+3]=255;}if(!marked)throw new Error('Paint a region or upload a mask first.');const canvas=document.createElement('canvas');canvas.width=paint.width;canvas.height=paint.height;canvas.getContext('2d').putImageData(pixels,0,0);return canvas.toDataURL('image/png');}
$('#mask-upload').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{const url=URL.createObjectURL(file);try{const image=await loadImage(url);if(image.naturalWidth!==paint.width||image.naturalHeight!==paint.height)throw new Error(`Mask must match the resized reference: ${paint.width} × ${paint.height}.`);ctx.clearRect(0,0,paint.width,paint.height);ctx.drawImage(image,0,0);const pixels=ctx.getImageData(0,0,paint.width,paint.height);for(let i=0;i<pixels.data.length;i+=4){const v=pixels.data[i];pixels.data[i]=255;pixels.data[i+1]=85;pixels.data[i+2]=100;pixels.data[i+3]=v;}ctx.putImageData(pixels,0,0);}finally{URL.revokeObjectURL(url);}formError();}catch(error){formError(error.message);}e.target.value='';};

async function api(path,options={}){const response=await fetch(path,{cache:'no-store',...options});const data=await response.json();if(!response.ok){const detail=data.detail||data.error;throw new Error(Array.isArray(detail)?detail.map(x=>x.msg).join('; '):detail||`HTTP ${response.status}`);}return data;}
async function refreshHealth(){try{const h=await api('/api/images/health');syncDwm(h);state.ready=!!h.ready;state.enhancementReady=!!h.prompt_enhancement?.t2i&&!!h.prompt_enhancement?.edit;$('#dot').className=state.ready?'ready':h.error?'error':'';$('#health').textContent=state.ready?'Ready':h.loading?'Loading model':'Offline';$('#health').title=h.error||'';}catch{state.ready=false;$('#health').textContent='Connecting';}$('#generate').disabled=!state.ready||!!state.job;$('#enhance').disabled=!state.ready||!state.enhancementReady||!!state.job;$('#enhance-hint').textContent=state.enhancementReady?'Preview before applying':'Enhancers loading';}
function displayUrl(output){return '/api/images/files/'+encodeURIComponent(output.name);}
function resultCard(output,container){const article=document.createElement('article');article.className='result-card';article.dataset.name=output.name;const image=document.createElement('img');image.src=displayUrl(output);image.className='checker';image.alt='Generated image';const actions=document.createElement('div');actions.className='result-actions';const meta=document.createElement('span');meta.textContent=`${output.width||''} × ${output.height||''} · seed ${output.seed}`;const right=document.createElement('div');const reuse=document.createElement('button');reuse.className='quiet';reuse.textContent='Add reference';reuse.onclick=()=>useAsReference(output.name);const link=document.createElement('a');link.href=image.src;link.download=output.name;link.textContent='Download PNG ↓';const remove=document.createElement('button');remove.className='quiet danger';remove.textContent='Delete';remove.onclick=async()=>{const item=galleryItems.find(i=>i.name===output.name);if(item)await trashImages([item.id]);else formError('Refresh the gallery and try again.');};right.append(reuse,link,remove);actions.append(meta,right);article.append(image,actions);container.append(article);}
async function useAsReference(name){
  await addGalleryReferences([name]);
}
async function generate(enhance=false){if(state.job||!state.ready)return;await referenceIngestion;formError();try{const prompt=$('#prompt').value.trim();if(!prompt)throw new Error('Enter a direction first.');const [width,height]=dimensions();if(width<256||height<256||width>4096||height>4096||width%32||height%32||width*height>4500000)throw new Error('Use dimensions from 256 to 4096 in multiples of 32, up to 4.5 megapixels.');if((enhance||$('#auto-enhance').checked)&&!state.enhancementReady)throw new Error('The official prompt enhancers are still installing. Turn off automatic enhancement to render now.');const seedText=$('#seed').value.trim();if(seedText&&!/^\d+$/.test(seedText))throw new Error('Seed must be a positive whole number.');const seed=seedText?Number(seedText):null;if(seed!==null&&!Number.isSafeInteger(seed))throw new Error('Seed is too large.');const refs=state.refs.map(r=>r.data);if(state.mode==='masked'&&refs.length>9)throw new Error('Mask editing allows nine references plus one mask.');if(['edit','extract','masked','annotate'].includes(state.mode)&&!refs.length)throw new Error('Add at least one reference image.');if(state.mode==='annotate'){const canvas=document.createElement('canvas');canvas.width=paint.width;canvas.height=paint.height;const context=canvas.getContext('2d');context.drawImage(await loadImage(refs[0]),0,0);context.drawImage(paint,0,0);refs[0]=canvas.toDataURL('image/png');}
const payload={prompt,enhance_prompt:!enhance&&$('#auto-enhance').checked,auto_aspect_ratio:!enhance&&$('#auto-enhance').checked&&$('#auto-ratio').checked,mode:state.mode,images_b64:refs,width,height,seed,num_inference_steps:Number($('#steps').value),n:Number($('#count').value),negative_prompt:$('#negative').value.trim(),true_cfg_scale:Number($('#guidance').value),dwm_scale:Number($('#dwm-scale').value),use_kv_cache:$('#kv-cache').checked,reference_resolution:Number($('#reference-detail').value),preserve_unmasked:$('#preserve').checked};if(state.mode==='masked')payload.mask_b64=maskData();if(payload.negative_prompt&&payload.true_cfg_scale<=1)throw new Error('Set guidance above 1 to use a negative prompt.');if(payload.true_cfg_scale>1&&!payload.negative_prompt)throw new Error('Add a negative prompt or set guidance to 1.');$('#generate').disabled=true;state.submittedVersion=state.inputVersion;const job=await api(enhance?'/api/images/enhance':'/api/images/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});state.job=job.id;localStorage.setItem('frosty.imageJob',job.id);$('#enhance').disabled=true;if(enhance){$('#enhance-preview').hidden=true;}else{$('#results').replaceChildren();$('#canvas-title').textContent=prompt;$('#gallery-section').hidden=true;}await watchJob();}catch(error){formError(error.message);$('#generate').disabled=!state.ready||!!state.job;}}
$('#generate').onclick=()=>generate();$('#enhance').onclick=()=>generate(true);document.addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();generate();}});
async function watchJob(){if(!state.job)return;$('#status-panel').hidden=false;try{const job=await api('/api/images/jobs/'+state.job);$('#job-stage').textContent=job.stage;$('#progress').max=job.total_steps||1;$('#progress').value=job.completed_steps||0;$('#elapsed').textContent=`${Math.floor(Date.now()/1000-job.created_at)} seconds elapsed`;
if(['done','error','cancelled'].includes(job.status)){state.job=null;localStorage.removeItem('frosty.imageJob');$('#spinner').hidden=true;$('#cancel').hidden=true;$('#generate').disabled=!state.ready;if(job.outputs?.length){$('#results').replaceChildren();job.outputs.forEach(o=>resultCard(o,$('#results')));}if(job.status==='error'){formError(job.error);$('#job-stage').textContent='Render stopped';}if(job.status==='done'){$('#elapsed').textContent=`Completed in ${job.seconds}s`;if(job.enhancement)showEnhancement(job.enhancement,job.kind==='image');}$('#enhance').disabled=!state.ready||!state.enhancementReady;await loadGallery();return;}$('#spinner').hidden=false;$('#cancel').hidden=false;setTimeout(watchJob,1500);}catch(error){formError(error.message);if(error.message==='Unknown image job'){state.job=null;localStorage.removeItem('frosty.imageJob');$('#generate').disabled=!state.ready;}else setTimeout(watchJob,5000);}}
$('#cancel').onclick=async()=>{if(!state.job)return;try{await api('/api/images/jobs/'+state.job+'/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});$('#job-stage').textContent='Cancelling after the current operation';}catch(error){formError(error.message);}};
async function loadGallery(){
  const version=++galleryRequest;
  try{
    const result=await api('/api/images/gallery');if(!result.ok)throw new Error(result.error||'Gallery unavailable');
    const trash=galleryTab==='trash'?await api('/api/images/gallery/trash'):null;
    if(version!==galleryRequest)return;
    galleryItems=result.items||[];$('#gallery-count').textContent=result.count;$('#trash-count').textContent=result.trash_count;
    shownItems=trash?trash.items||[]:galleryItems;
    const valid=new Set(shownItems.map(item=>item.id));selectedGallery=new Set([...selectedGallery].filter(id=>valid.has(id)));
    renderGallery();
  }catch(error){$('#gallery-error').textContent=error.message;}
}
$('#gallery-toggle').onclick=()=>{const section=$('#gallery-section');section.hidden=!section.hidden;if(!section.hidden){loadGallery();section.scrollIntoView({behavior:'smooth'});}};$('#refresh-gallery').onclick=loadGallery;
let referenceIngestion=Promise.resolve(),galleryItems=[],shownItems=[],selectedGallery=new Set(),galleryTab='photos',galleryRequest=0,libraryBusy=false,lastTrash=[];
state.inputVersion=0;state.dwmInitialized=false;state.dwmTouched=false;
updateDimensions();setMode('auto');renderRefs();refreshHealth();loadGallery();setInterval(refreshHealth,12000);
fetch('/api/workspaces').then(r=>r.json()).then(w=>{$('#video-workspace').hidden=!w.video;}).catch(()=>{});
const savedJob=localStorage.getItem('frosty.imageJob');if(savedJob&&/^img_[a-f0-9]{24}$/.test(savedJob)){state.job=savedJob;watchJob();}

function showEnhancement(result,used=false){state.enhancement=result;$('#use-enhanced').disabled=state.submittedVersion!==state.inputVersion;$('#enhanced-prompt').value=result.prompt;$('#enhance-preview').hidden=false;$('#enhance-model').textContent=used?'Prompt used for this image · edit and apply to reuse':result.task==='edit'?'Qwen editing enhancer · original direction kept until you apply':'Qwen creation enhancer · original direction kept until you apply';let ratio=result.wh_ratio;if(!sizes[ratio]&&result.ratio_follow){const match=result.ratio_follow.match(/^<image(\d+)>$/);const ref=match&&state.refs[Number(match[1])-1];if(ref)ratio=Object.keys(sizes).reduce((best,key)=>Math.abs(sizes[key][0]/sizes[key][1]-ref.width/ref.height)<Math.abs(sizes[best][0]/sizes[best][1]-ref.width/ref.height)?key:best,'1:1');}state.enhancement.suggestedRatio=sizes[ratio]?ratio:null;$('#enhance-ratio-row').hidden=!state.enhancement.suggestedRatio;$('#enhance-ratio').checked=false;$('#enhance-ratio-label').textContent=`Also use suggested aspect ratio ${ratio||''}`;$('#enhanced-prompt').focus();}
$('#use-enhanced').onclick=()=>{const value=$('#enhanced-prompt').value.trim();if(!value){formError('The enhanced direction is empty.');return;}$('#prompt').value=value;$('#auto-enhance').checked=false;$('#auto-ratio').checked=false;if($('#enhance-ratio').checked&&state.enhancement?.suggestedRatio){$('#ratio').value=state.enhancement.suggestedRatio;updateDimensions();}$('#enhance-preview').hidden=true;formError();};
$('#discard-enhanced').onclick=()=>{$('#enhance-preview').hidden=true;state.enhancement=null;};

function invalidateEnhancement(){state.inputVersion=(state.inputVersion||0)+1;if($('#enhance-preview'))$('#enhance-preview').hidden=true;state.enhancement=null;}
$('#prompt').addEventListener('input',invalidateEnhancement);
function referencesChanged(){invalidateEnhancement();ctx.clearRect(0,0,paint.width,paint.height);setMode(state.mode);renderRefs();}
function ingestReferences(loader){
  referenceIngestion=referenceIngestion.then(async()=>{
    const added=await loader(),limit=state.mode==='masked'?9:10;
    if(state.refs.length+added.length>limit)throw new Error(`Use up to ${limit} references${state.mode==='masked'?' plus one mask':''}.`);
    state.refs.push(...added);referencesChanged();
  }).catch(error=>{formError(error.message);});return referenceIngestion;
}
function previewReference(ref,index){$('#reference-preview-image').src=ref.data;$('#reference-preview-caption').textContent=`Image ${index+1} · ${ref.name} · ${ref.width} × ${ref.height}`;$('#reference-preview').showModal();}
$('#close-reference-preview').onclick=()=>$('#reference-preview').close();
function replaceReference(index){
  const original=state.refs[index],picker=document.createElement('input');picker.type='file';picker.accept='image/png,image/jpeg,image/webp';
  picker.onchange=()=>{if(!picker.files[0])return;referenceIngestion=referenceIngestion.then(async()=>{const ref=await readReference(picker.files[0]),position=state.refs.indexOf(original);if(position>=0){state.refs[position]=ref;referencesChanged();}}).catch(error=>formError(error.message));};picker.click();
}
$('#clear-references').onclick=()=>{state.refs=[];referencesChanged();};
$('#add-gallery').onclick=()=>{galleryTab='photos';selectedGallery.clear();$('#gallery-section').hidden=false;loadGallery();$('#gallery-section').scrollIntoView({behavior:'smooth'});};
async function addGalleryReferences(names){
  await ingestReferences(async()=>{const refs=[];for(const name of names){const response=await fetch('/api/images/files/'+encodeURIComponent(name));if(!response.ok)throw new Error('Saved image is unavailable; refresh the gallery.');const blob=await response.blob();refs.push(await readReference(new File([blob],name,{type:blob.type||'image/png'})));}return refs;});
}
function syncDwm(health){
  const enabled=!!health.dwm?.enabled,select=$('#dwm-scale');
  if(!state.dwmInitialized&&!state.dwmTouched){const value=String(health.dwm_default_scale??0);if(![...select.options].some(o=>o.value===value))select.add(new Option(value,value));select.value=value;state.dwmInitialized=true;}
  select.disabled=!enabled;$('#dwm-status').textContent=enabled?'Active on the image encoder · saved with every image':'No DWM profile loaded · clean image encoder';
}
$('#dwm-scale').onchange=()=>{state.dwmTouched=true;};
function updateSelection(){
  $('#selection-count').textContent=`${selectedGallery.size} selected`;
  for(const id of ['add-selected','delete-selected','restore-selected'])$('#'+id).disabled=libraryBusy||!selectedGallery.size;
  $('#add-selected').hidden=$('#delete-selected').hidden=galleryTab==='trash';$('#restore-selected').hidden=galleryTab!=='trash';
  $('#select-all-gallery').checked=!!shownItems.length&&selectedGallery.size===shownItems.length;
  $('#select-all-gallery').indeterminate=selectedGallery.size>0&&selectedGallery.size<shownItems.length;
  $('#select-all-gallery').disabled=libraryBusy;
}
function renderGallery(){
  const root=$('#gallery');root.replaceChildren();
  for(const [id,tab] of [['show-library','photos'],['show-trash','trash']]){$('#'+id).classList.toggle('selected',galleryTab===tab);$('#'+id).setAttribute('aria-pressed',String(galleryTab===tab));}
  if(!shownItems.length){const note=document.createElement('p');note.className='hint';note.textContent=galleryTab==='trash'?'Trash is empty. Deleted photos stay recoverable here.':'Your finished images will appear here.';root.append(note);}
  for(const item of shownItems){
    const article=document.createElement('article');article.className='gallery-card'+(selectedGallery.has(item.id)?' selected':'');
    const selection=document.createElement('label');selection.className='selection';const check=document.createElement('input');check.type='checkbox';check.checked=selectedGallery.has(item.id);check.disabled=libraryBusy;check.setAttribute('aria-label','Select '+item.name);
    check.onchange=()=>{check.checked?selectedGallery.add(item.id):selectedGallery.delete(item.id);article.classList.toggle('selected',check.checked);updateSelection();};
    const label=document.createElement('span');label.textContent=new Date((item.deleted_at||item.created_at)*1000).toLocaleString();selection.append(check,label);article.append(selection);
    if(galleryTab==='photos'){
      const image=document.createElement('img');image.src=item.file_url;image.loading='lazy';image.className='checker';image.alt=item.prompt||'Saved image';image.tabIndex=0;
      image.onclick=()=>{$('#results').replaceChildren();resultCard(item,$('#results'));$('#canvas-title').textContent=item.prompt||item.name;$('#results').scrollIntoView({behavior:'smooth'});};image.onkeydown=e=>{if(e.key==='Enter')image.click();};article.append(image);
    }else{const placeholder=document.createElement('div');placeholder.className='trash-placeholder';placeholder.textContent=item.name+(item.state==='trashed'?'':' · recovery pending');article.append(placeholder);}
    const info=document.createElement('div');info.className='gallery-info';const prompt=document.createElement('p');prompt.textContent=item.prompt||item.name;info.append(prompt);
    if(galleryTab==='photos'){const meta=document.createElement('small');meta.textContent=`${item.width||'?'} × ${item.height||'?'} · seed ${item.seed??'unknown'}${item.dwm_scale!=null?' · DWM '+item.dwm_scale:''}`;info.append(meta);}
    const actions=document.createElement('div');actions.className='card-actions';
    const button=(text,action,danger=false)=>{const b=document.createElement('button');b.className='quiet'+(danger?' danger':'');b.textContent=text;b.disabled=libraryBusy;b.onclick=action;actions.append(b);};
    if(galleryTab==='photos'){button('Add reference',()=>addGalleryReferences([item.name]));const download=document.createElement('a');download.className='quiet';download.href=item.file_url;download.download=item.name;download.textContent='PNG ↓';actions.append(download);button('Delete',()=>trashImages([item.id]),true);}
    else button('Restore',()=>restoreImages([item.id]));
    info.append(actions);article.append(info);root.append(article);
  }updateSelection();
}
function notice(message,undo=[]){lastTrash=undo;$('#toast-message').textContent=message;$('#undo-trash').hidden=!undo.length;$('#toast').hidden=false;}
async function changeLibrary(route,ids){
  const result=await api('/api/images/gallery/'+route,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids})});
  const good=(result.results||[]).filter(row=>row.ok),bad=(result.results||[]).filter(row=>!row.ok);
  $('#gallery-error').textContent=bad.map(row=>row.error).join('\n');return good;
}
async function trashImages(ids){
  if(libraryBusy||!ids.length)return;libraryBusy=true;renderGallery();
  try{const good=await changeLibrary('trash',ids);const names=new Set(good.map(item=>item.name));$$('.result-card').filter(card=>names.has(card.dataset.name)).forEach(card=>card.remove());if(good.length)notice(`${good.length} photo${good.length===1?'':'s'} moved to Trash`,good.map(item=>item.trash_id));selectedGallery.clear();await loadGallery();}
  catch(error){$('#gallery-error').textContent=error.message;}
  finally{libraryBusy=false;renderGallery();}
}
async function restoreImages(ids){
  if(libraryBusy||!ids.length)return;libraryBusy=true;renderGallery();
  try{const good=await changeLibrary('restore',ids);if(good.length)notice(`${good.length} photo${good.length===1?'':'s'} restored`);selectedGallery.clear();await loadGallery();}
  catch(error){$('#gallery-error').textContent=error.message;}
  finally{libraryBusy=false;renderGallery();}
}
$('#show-library').onclick=()=>{galleryTab='photos';selectedGallery.clear();$('#gallery-error').textContent='';loadGallery();};
$('#show-trash').onclick=()=>{galleryTab='trash';selectedGallery.clear();$('#gallery-error').textContent='';loadGallery();};
$('#select-all-gallery').onchange=e=>{selectedGallery=e.target.checked?new Set(shownItems.map(item=>item.id)):new Set();renderGallery();};
$('#delete-selected').onclick=()=>trashImages([...selectedGallery]);$('#restore-selected').onclick=()=>restoreImages([...selectedGallery]);
$('#add-selected').onclick=()=>addGalleryReferences(shownItems.filter(item=>selectedGallery.has(item.id)).map(item=>item.name));
$('#undo-trash').onclick=()=>restoreImages(lastTrash.slice());$('#dismiss-toast').onclick=()=>{$('#toast').hidden=true;};
