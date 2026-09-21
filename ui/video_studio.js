
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
fetch('/api/workspaces').then(r=>r.json()).then(w=>{$('#image-workspace').hidden=!w.image;$('#choose-image-library').hidden=!w.image;}).catch(()=>{});
let imageData=null,imageName='',engines=[],selectedEngineId='',appliedEngineConfiguration='';
const capNames={text_to_video:'Text',image_to_video:'Image',same_face:'Same Face',native_audio:'Audio',scene_lab:'Scene Lab'};
const selectedEngine=()=>engines.find(x=>x.id===selectedEngineId)||engines[0]||null;
const hasCap=name=>!!selectedEngine()?.capabilities?.includes(name);
const jsonPromptTemplate={
  version:"1.0",
  summary:"A cinematic tracking shot of a red fox moving through a snowy pine forest at dawn.",
  intent:{mood:"quiet, alert, immersive",priority:"natural motion, coherent anatomy, and synchronized environmental sound"},
  subjects:[{
    id:"fox_1",
    type:"adult red fox",
    appearance:{fur:"rich red-orange winter coat",markings:"white chest and dark lower legs",condition:"healthy and dry"},
    action:"trot steadily along a narrow snow-covered trail, occasionally turning its ears toward distant sounds",
    performance:"natural animal behavior; calm but attentive",
    continuity:"preserve coat markings, scale, and body proportions for the entire shot"
  }],
  environment:{
    location:"dense alpine pine forest",
    time_of_day:"early dawn",
    weather:"cold, still air with very light drifting snow",
    ground:"fresh powder over a narrow forest trail",
    background:"layered pine trunks fading into pale blue atmospheric haze",
    physical_details:["small puffs of powder under each paw", "subtle breath vapor", "occasional snow falling from branches"]
  },
  timeline:[
    {time:"0.0-1.5s",visual_action:"begin on a wide profile view as the fox enters frame left",camera_action:"smooth lateral tracking begins",audio_event:"soft paw impacts and quiet winter wind"},
    {time:"1.5-3.8s",visual_action:"fox crosses the center of frame and briefly looks toward camera without stopping",camera_action:"ease into a medium profile framing with stable horizon",audio_event:"nearby branch creak and a distant bird call on the right"},
    {time:"3.8-5.2s",visual_action:"fox continues toward a brighter opening between the trees",camera_action:"gently lag behind and hold the final composition",audio_event:"paw sounds recede into natural forest ambience"}
  ],
  camera:{
    shot_design:"single continuous shot",
    framing:"wide-to-medium side profile",
    lens:"35mm spherical lens",
    movement:"low, stabilized lateral tracking at the fox's pace",
    height:"approximately 45 cm above the ground",
    focus:"continuous focus on the fox's eyes and head",
    depth_of_field:"moderate; subject sharp with softly layered background",
    composition:"fox travels left to right with open lead room",
    prohibited_moves:["no cuts", "no sudden zoom", "no orbit", "no handheld shake"]
  },
  lighting:{
    key:"soft cool dawn skylight",
    accent:"faint warm sunlight filtering between distant trees",
    contrast:"gentle and realistic",
    exposure:"retain detail in white snow and dark fur"
  },
  visual_style:{
    medium:"photoreal cinematic nature footage",
    color_palette:["snow white", "pine green", "cool blue shadows", "warm red fur"],
    texture:"fine fur, powder snow, and bark detail without oversharpening",
    motion_rendering:"natural shutter motion blur",
    grading:"restrained documentary color grade"
  },
  audio:{
    dialogue:[],
    ambience:"quiet stereo winter forest with light wind through pine needles",
    foreground_sounds:["soft rhythmic paw impacts in snow", "subtle fox breathing"],
    background_sounds:["one distant bird call from the right", "occasional branch creak"],
    music:"none",
    mix:"natural dynamic range; ambience remains below foreground movement",
    synchronization:"paw impacts and visible environmental motion must align precisely"
  },
  continuity:["one fox only", "consistent direction of travel", "stable weather and dawn lighting", "no unexplained objects entering the shot"],
  avoid:["anatomy distortion", "extra limbs or tails", "sliding feet", "flicker", "warped trees", "text", "logos", "subtitles", "music"]
};
let promptMode='text';
const promptDrafts={text:$('#p').value,json:JSON.stringify(jsonPromptTemplate,null,2)};
function parsedJsonPrompt(){let value;try{value=JSON.parse($('#p').value);}catch(error){throw new Error('Invalid JSON: '+error.message);}if(!value||Array.isArray(value)||typeof value!=='object')throw new Error('JSON prompt must be a non-empty object.');if(!Object.keys(value).length)throw new Error('JSON prompt cannot be empty.');return value;}
function promptForRequest(){if(promptMode==='json')return JSON.stringify(parsedJsonPrompt(),null,2);return $('#p').value.trim();}
function updatePromptNote(){const note=$('#prompt-note');if(promptMode!=='json'){note.className='sub prompt-note';note.textContent='Natural-language direction sent directly to the selected model.';return;}try{const value=parsedJsonPrompt(),chars=JSON.stringify(value,null,2).length;note.className='sub prompt-note json-valid';note.textContent='Valid JSON object · '+Object.keys(value).length+' top-level fields · '+chars.toLocaleString()+' characters. The model receives this structure as prompt text.';}catch(error){note.className='sub prompt-note json-invalid';note.textContent=error.message;}}
function setPromptMode(next){promptDrafts[promptMode]=$('#p').value;promptMode=next;$('#p').value=promptDrafts[promptMode];$('#p').classList.toggle('json-prompt',promptMode==='json');$('#p').placeholder=promptMode==='json'?'Enter one valid JSON object…':'Describe the subject, action, camera, lighting, dialogue, and sound…';$('#json-template').hidden=promptMode!=='json';$('#json-format').hidden=promptMode!=='json';updatePromptNote();}
$('#prompt-format').onchange=()=>setPromptMode($('#prompt-format').value);
$('#json-template').onclick=()=>{if($('#p').value.trim()&&!confirm('Replace the current JSON with the detailed template?'))return;$('#p').value=JSON.stringify(jsonPromptTemplate,null,2);promptDrafts.json=$('#p').value;updatePromptNote();};
$('#json-format').onclick=()=>{try{$('#p').value=JSON.stringify(parsedJsonPrompt(),null,2);promptDrafts.json=$('#p').value;updatePromptNote();}catch(error){updatePromptNote();$('#status').innerHTML='<span class="err">'+esc(error.message)+'</span>';}};
$('#p').addEventListener('input',()=>{promptDrafts[promptMode]=$('#p').value;updatePromptNote();});
$('#p').addEventListener('keydown',event=>{if(promptMode==='json'&&event.key==='Tab'){event.preventDefault();const start=event.target.selectionStart;event.target.setRangeText('  ',start,event.target.selectionEnd,'end');promptDrafts.json=event.target.value;updatePromptNote();}});
function setOptions(select,values,labels={}){const current=select.value;select.innerHTML='';(values||[]).forEach(value=>{const option=document.createElement('option');option.value=value;option.textContent=labels[value]||value;select.appendChild(option);});if([...select.options].some(x=>x.value===current))select.value=current;}
function settingsPayload(){const engine=selectedEngine(),controls=engine?.controls||{},payload={};
  if(controls.resolutions?.length)payload.resolution=$('#resolution').value;if(controls.aspect_ratios?.length)payload.aspect_ratio=$('#aspect-ratio').value;
  if(controls.steps)payload.num_inference_steps=Number($('#steps').value);if(controls.guidance)payload.guidance_scale=Number($('#guidance').value);
  if(controls.negative_prompt&&$('#negative-prompt').value.trim())payload.negative_prompt=$('#negative-prompt').value.trim();return payload;}
function applyEngine(){const engine=selectedEngine();if(!engine)return;const controls=engine.controls||{},health=engine.health||{};
  $('#engine-title').textContent=engine.label;$('#engine-description').textContent=engine.description;$('#lab-engine-badge').textContent=engine.label;
  $('#engine-caps').innerHTML='';engine.capabilities.forEach(name=>{const chip=document.createElement('span');chip.className='cap on';chip.textContent=capNames[name]||name.replaceAll('_',' ');$('#engine-caps').appendChild(chip);});
  $('#dot').className='dot '+(health.ready?'up':'down');$('#ep').textContent=health.ready?'ready':health.loading||health.status==='loading'?'loading model…':'model offline';
  const configuration=JSON.stringify([engine.id,controls,engine.capabilities]);if(configuration===appliedEngineConfiguration)return;appliedEngineConfiguration=configuration;
  const durations=(controls.durations||[5]).map(Number),durationLabels=Object.fromEntries(durations.map(n=>[n,n+' seconds']));setOptions($('#duration'),durations,durationLabels);
  $('#duration-badge').textContent=durations.length?(Math.min(...durations)+'–'+Math.max(...durations)+' sec'):'model default';
  if(typeof scenes!=='undefined'){scenes.forEach(scene=>{if(!durations.includes(Number(scene.duration)))scene.duration=durations[0];});renderScenes();}
  setOptions($('#resolution'),controls.resolutions||[]);setOptions($('#aspect-ratio'),controls.aspect_ratios||[]);
  $('#resolution-field').hidden=!controls.resolutions?.length;$('#aspect-field').hidden=!controls.aspect_ratios?.length;
  $('#steps-field').hidden=!controls.steps;$('#guidance-field').hidden=!controls.guidance;$('#negative-field').hidden=!controls.negative_prompt;
  $('#quality-field').hidden=!controls.steps;if(controls.steps){$('#steps').min=controls.steps.min;$('#steps').max=controls.steps.max;$('#steps').value=Math.max(controls.steps.min,Math.min(controls.steps.max,Number($('#steps').value)||controls.steps.default));}
  if(controls.guidance){$('#guidance').min=controls.guidance.min;$('#guidance').max=controls.guidance.max;$('#guidance').step=controls.guidance.step||.5;if(!$('#guidance').value)$('#guidance').value=controls.guidance.default;}
  $('#settings-note').hidden=!!(controls.resolutions?.length||controls.aspect_ratios?.length||controls.steps||controls.guidance||controls.negative_prompt);
  if(!hasCap('same_face')){$('#lock').checked=false;$('#lab-lock').checked=false;$('#continuity').disabled=false;}$('#lock-help').textContent=hasCap('same_face')?'Use the image as an identity reference instead of the opening frame.':'This model uses the image as an opening-frame guide; identity lock is unavailable.';
  $('#lab-lock-help').textContent=hasCap('same_face')?'Reference the same face in every scene.':'Same Face is unavailable for this model.';renderImage();}
async function refreshEngines(){try{const previous=selectedEngineId||$('#engine-select').value,r=await fetch('/api/engines',{cache:'no-store'}),j=await r.json();if(!j.ok)throw new Error(j.error||'Could not load models');engines=(j.engines||[]).filter(x=>x.capabilities?.includes("text_to_video"));selectedEngineId=engines.some(x=>x.id===previous)?previous:(engines.some(x=>x.id===j.default)?j.default:engines[0]?.id||'');
  const select=$('#engine-select');select.innerHTML='';engines.forEach(engine=>{const option=document.createElement('option');option.value=engine.id;const state=engine.health?.ready?'●':engine.health?.loading||engine.health?.status==='loading'?'◌':'○';option.textContent=state+' '+engine.label;select.appendChild(option);});select.value=selectedEngineId;applyEngine();}
  catch(e){$('#dot').className='dot down';$('#ep').textContent='model menu offline';$('#engine-description').textContent=e.message||e;}}

$('#engine-select').onchange=()=>{selectedEngineId=$('#engine-select').value;applyEngine();};
$('#settings-toggle').onclick=()=>{const panel=$('#settings-panel'),willOpen=panel.hidden;panel.hidden=!willOpen;$('#settings-toggle').setAttribute('aria-expanded',String(willOpen));$('#settings-toggle').textContent=willOpen?'Close settings':'Settings';};

$$('.tab').forEach(b=>b.onclick=()=>{$$('.tab').forEach(x=>x.classList.toggle('active',x===b));$$('.panel').forEach(x=>x.classList.remove('active'));$('#'+b.dataset.tab+'-panel').classList.add('active');if(b.dataset.tab==='gallery')loadGallery();if(b.dataset.tab==='jobs')refreshJobs();});

function imageFile(file){return new Promise((resolve,reject)=>{if(!file||!file.type.startsWith('image/'))return reject(new Error('Choose an image file.'));
  const url=URL.createObjectURL(file),img=new Image();img.onload=()=>{URL.revokeObjectURL(url);let w=img.naturalWidth,h=img.naturalHeight,max=1600;
    if(Math.max(w,h)>max){const scale=max/Math.max(w,h);w=Math.round(w*scale);h=Math.round(h*scale);}const c=document.createElement('canvas');c.width=w;c.height=h;
    c.getContext('2d').drawImage(img,0,0,w,h);resolve(c.toDataURL('image/jpeg',.92));};img.onerror=()=>{URL.revokeObjectURL(url);reject(new Error('That image could not be read.'));};img.src=url;});}
async function setImage(file){try{imageData=await imageFile(file);imageName=file.name;renderImage();}catch(e){alert(e.message);}}
function renderImage(){const canImage=hasCap('image_to_video'),canLock=hasCap('same_face');$$('.image-drop').forEach(drop=>{const empty=drop.querySelector('.empty'),preview=drop.querySelector('.preview');drop.classList.toggle('disabled',!canImage);
  if(imageData){empty.hidden=true;preview.hidden=false;preview.innerHTML='<img src="'+imageData+'"><div class="shade">'+esc(imageName)+'</div><button class="remove-image" title="Remove image" type="button">×</button>';preview.querySelector('button').onclick=e=>{e.stopPropagation();imageData=null;imageName='';$('#lock').checked=false;$('#lab-lock').checked=false;renderImage();};}
  else{empty.hidden=false;preview.hidden=true;preview.innerHTML='';}});$$('#lock,#lab-lock').forEach(x=>x.disabled=!imageData||!canLock);}
$$('.image-drop').forEach((drop,i)=>{const input=$$('.image-input')[i];drop.onclick=()=>{if(hasCap('image_to_video'))input.click();};input.onchange=()=>input.files[0]&&setImage(input.files[0]);
  drop.ondragover=e=>{e.preventDefault();drop.classList.add('drag')};drop.ondragleave=()=>drop.classList.remove('drag');drop.ondrop=e=>{e.preventDefault();drop.classList.remove('drag');e.dataTransfer.files[0]&&setImage(e.dataTransfer.files[0]);};});
document.addEventListener('paste',e=>{const f=[...(e.clipboardData?.files||[])].find(x=>x.type.startsWith('image/'));if(f)setImage(f);});
$('#lab-lock').onchange=()=>{$('#continuity').disabled=$('#lab-lock').checked;if($('#lab-lock').checked)$('#continuity').checked=false;};

$('#rnd').onclick=()=>{$('#s').value=Math.floor(Math.random()*2**31);saveDraft();};
function showResult(container,url,seed,mode,sceneUrls=[]){container.innerHTML='';const v=document.createElement('video');v.src=url;v.controls=true;v.autoplay=true;v.loop=sceneUrls.length===0;container.appendChild(v);
  const tools=document.createElement('div');tools.className='result-tools';tools.innerHTML='<span class="badge">'+esc(mode)+'</span><a class="dl" href="'+url+'" download="frostyvl_'+esc(seed)+'.mp4">Download video ↓</a>';container.appendChild(tools);
  if(sceneUrls.length){const clips=document.createElement('div');clips.className='scene-clips';sceneUrls.forEach((u,i)=>{const sv=document.createElement('video');sv.src=u;sv.controls=true;sv.muted=true;sv.title='Scene '+(i+1);clips.appendChild(sv);});container.appendChild(clips);}}
function formatBytes(bytes){const n=Number(bytes)||0;if(n<1024)return n+' B';if(n<1024**2)return (n/1024).toFixed(1)+' KB';if(n<1024**3)return (n/1024**2).toFixed(1)+' MB';return (n/1024**3).toFixed(1)+' GB';}
function formatDate(seconds){const d=new Date(Number(seconds)*1000);return Number.isNaN(d.valueOf())?'':d.toLocaleString([], {dateStyle:'medium',timeStyle:'short'});}
let galleryItems=[],trashView=false,selectedVideos=new Set(),lastTrashed=[];
async function api(path,body){const response=await fetch(path,{cache:'no-store',...(body!==undefined?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{})});const data=await response.json();if(!response.ok)throw new Error(data.error||data.detail||'Request failed');return data;}
function filteredVideos(){const q=$('#gallery-search').value.toLowerCase();return galleryItems.filter(x=>[x.prompt,x.name,x.engine_label].join(' ').toLowerCase().includes(q));}
function updateSelection(){const visible=filteredVideos();$('#select-all-videos').checked=!!visible.length&&visible.every(x=>selectedVideos.has(x.id));$('#delete-videos').disabled=!selectedVideos.size;$('#restore-videos').disabled=!selectedVideos.size;}
function reuseVideo(item){
  if(item.engine_id&&engines.some(x=>x.id===item.engine_id)){selectedEngineId=item.engine_id;$('#engine-select').value=selectedEngineId;applyEngine();}
  promptMode=item.prompt_format==='json'?'json':'text';$('#prompt-format').value=promptMode;$('#p').value=item.prompt||'';promptDrafts[promptMode]=$('#p').value;
  $('#p').classList.toggle('json-prompt',promptMode==='json');$('#json-template').hidden=promptMode!=='json';$('#json-format').hidden=promptMode!=='json';updatePromptNote();
  $('#s').value=item.seed??'';
  for(const [key,id] of [['duration_seconds','duration'],['resolution','resolution'],['aspect_ratio','aspect-ratio'],['num_inference_steps','steps'],['guidance_scale','guidance'],['negative_prompt','negative-prompt']])if(item[key]!=null)$('#'+id).value=item[key];
  imageData=null;imageName='';$('#lock').checked=false;$('#lab-lock').checked=false;renderImage();
  document.querySelector('[data-tab="create"]').click();$('#status').textContent='Settings reused. Reattach the original reference image if needed.';saveDraft();
}
function renderGallery(){const root=$('#gallery'),items=filteredVideos();root.innerHTML='';$('#gallery-summary').textContent=items.length+' '+(trashView?'items in Trash':'saved videos');$('#delete-videos').hidden=trashView;$('#restore-videos').hidden=!trashView;$('#video-trash').textContent=trashView?'Back to videos':'Trash';
  if(!items.length){root.innerHTML='<div class="empty-gallery"><b>'+ (trashView?'Trash is empty':'Your videos will stay here')+'</b><div class="sub">'+(trashView?'Deleted videos can be restored here.':'Completed clips and movies appear here. Try a different search if needed.')+'</div></div>';}
  items.forEach(item=>{const tile=document.createElement('article');tile.className='tile';const selection=document.createElement('label');selection.className='tile-select';const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.checked=selectedVideos.has(item.id);checkbox.setAttribute('aria-label','Select '+item.name);checkbox.onchange=()=>{checkbox.checked?selectedVideos.add(item.id):selectedVideos.delete(item.id);updateSelection();};const filename=document.createElement('span');filename.textContent=item.name;filename.title=item.name;selection.append(checkbox,filename);tile.append(selection);
    if(!trashView){const video=document.createElement('video');video.src=item.file_url;video.controls=true;video.muted=true;video.playsInline=true;video.preload='metadata';tile.append(video);}
    const meta=document.createElement('div');meta.className='meta';const title=document.createElement('b');title.textContent=trashView?'Deleted '+formatDate(item.deleted_at):[item.engine_label,item.mode,item.seed!=null?'seed '+item.seed:''].filter(Boolean).join(' · ');const prompt=document.createElement('div');prompt.textContent=item.prompt||item.name;
    const tools=document.createElement('div');tools.className='tile-tools';
    if(!trashView){const link=document.createElement('a');link.className='dl';link.href=item.file_url;link.download=item.name;link.textContent='Download';const reuse=document.createElement('button');reuse.className='ghost';reuse.textContent='Reuse settings';reuse.onclick=()=>reuseVideo(item);tools.append(link,reuse);}
    const action=document.createElement('button');action.className=trashView?'ghost':'danger';action.textContent=trashView?'Restore':'Delete';action.onclick=()=>libraryAction(trashView?'restore':'trash',[item.id]);tools.append(action);meta.append(title,prompt,tools);tile.append(meta);root.append(tile);
  });updateSelection();}
async function loadGallery(){try{const j=await api('/api/videos/gallery'+(trashView?'/trash':''));if(j.ok===false)throw new Error(j.error||'Gallery unavailable');galleryItems=j.items||[];selectedVideos=new Set([...selectedVideos].filter(id=>galleryItems.some(x=>x.id===id)));renderGallery();}catch(e){$('#gallery-summary').textContent='Gallery unavailable';$('#gallery-message').textContent=e.message;}}
async function libraryAction(action,ids){try{const j=await api('/api/videos/gallery/'+action,{ids});const good=(j.results||[]).filter(x=>x.ok),bad=(j.results||[]).filter(x=>!x.ok);if(action==='trash'&&good.length){lastTrashed=good.map(x=>x.trash_id);$('#undo-video-trash').hidden=false;}$('#gallery-message').textContent=good.length+' '+(action==='trash'?'moved to Trash':'restored')+(bad.length?' · '+bad.map(x=>x.error).join('; '):'');selectedVideos.clear();await loadGallery();}catch(e){$('#gallery-message').textContent=e.message;}}
$('#gallery-refresh').onclick=loadGallery;$('#gallery-search').oninput=()=>{selectedVideos.clear();renderGallery();};$('#video-trash').onclick=()=>{trashView=!trashView;selectedVideos.clear();loadGallery();};
$('#select-all-videos').onchange=()=>{selectedVideos=new Set($('#select-all-videos').checked?filteredVideos().map(x=>x.id):[]);renderGallery();};
$('#delete-videos').onclick=()=>libraryAction('trash',[...selectedVideos]);$('#restore-videos').onclick=()=>libraryAction('restore',[...selectedVideos]);
$('#undo-video-trash').onclick=async()=>{await libraryAction('restore',lastTrashed);$('#undo-video-trash').hidden=true;};

let watchedJobs={clip:null,scenes:null},seenResults=new Set(),jobsRefreshing=false;
try{watchedJobs=JSON.parse(localStorage.getItem('frosty.video.jobs')||'null')||watchedJobs;}catch{}
function watchJob(kind,id){watchedJobs[kind]=id;try{localStorage.setItem('frosty.video.jobs',JSON.stringify(watchedJobs));}catch{}}
async function cancelVideo(id){try{await api('/api/videos/jobs/'+encodeURIComponent(id)+'/cancel',{});await refreshJobs();}catch(e){$('#status').textContent=e.message;}}
async function refreshJobs(){if(jobsRefreshing)return;jobsRefreshing=true;try{const data=await api('/api/videos/jobs'),jobs=data.items||[];$('#jobs-count').textContent=jobs.filter(j=>!['done','error','cancelled'].includes(j.status)).length;const root=$('#video-jobs');root.innerHTML='';
  if(!jobs.length)root.textContent='No jobs yet. Queue a clip or a movie to get started.';
  jobs.forEach(job=>{const card=document.createElement('article');card.className='job-card';const copy=document.createElement('div');copy.className='job-copy';const title=document.createElement('b');title.textContent=(job.kind==='scenes'?'Movie':'Clip')+' · '+(job.engine_label||job.engine_id);const note=document.createElement('small');note.textContent=(job.prompt||'').slice(0,180)+' · '+(job.error||job.stage);copy.append(title,note);const badge=document.createElement('span');badge.className='badge';badge.textContent=job.status;card.append(copy,badge);
    if(!['done','error','cancelled'].includes(job.status)){const cancel=document.createElement('button');cancel.className='ghost';cancel.textContent=job.status==='queued'?'Cancel queued':'Stop after render';cancel.disabled=job.cancel_requested;cancel.onclick=()=>cancelVideo(job.id);card.append(cancel);}
    if(job.file_url){const link=document.createElement('a');link.className='dl';link.href=job.file_url;link.download='frosty-video.mp4';link.textContent='Download';card.append(link);}root.append(card);
  });
  for(const kind of ['clip','scenes']){const job=jobs.find(j=>j.id===watchedJobs[kind]);if(!job){if(watchedJobs[kind])$(kind==='clip'?'#status':'#lab-status').textContent='Previous job is no longer in this Studio session. Check the gallery for saved results.';continue;}const target=kind==='clip'?'#status':'#lab-status';$(target).textContent=job.error||job.stage+(job.used_seed!=null?' · seed '+job.used_seed:'');
    if(kind==='scenes'){$('#lab-progress').hidden=false;$('#lab-progress span').style.width=(job.status==='done'?100:100*(job.completed_scenes||0)/(job.total_scenes||1))+'%';}
    if(job.file_url&&!seenResults.has(job.id)){seenResults.add(job.id);showResult($(kind==='clip'?'#result':'#lab-result'),job.file_url,job.used_seed,job.mode||job.kind,job.scene_urls||[]);loadGallery();}
  }
}catch(e){$('#video-jobs').textContent='Could not refresh jobs: '+e.message;}finally{jobsRefreshing=false;}}
async function generate(){const engine=selectedEngine();try{const prompt=promptForRequest();if(!prompt)throw new Error('Enter a direction first.');if(!engine?.health?.ready)throw new Error('The selected model is offline. Your draft is saved.');const payload={kind:'clip',request_id:crypto.randomUUID(),prompt,prompt_format:promptMode,engine_id:engine.id,seed:$('#s').value.trim(),duration_seconds:Number($('#duration').value),identity_lock:$('#lock').checked,...settingsPayload()};if(imageData)payload.image_b64=imageData;$('#go').disabled=true;const job=await api('/api/videos/jobs',payload);watchJob('clip',job.id);$('#status').textContent='Clip queued';saveDraft();await refreshJobs();}catch(e){$('#status').textContent=e.message;}finally{$('#go').disabled=false;}}
$('#go').onclick=generate;$('#p').addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key==='Enter')generate();});$('#jobs-refresh').onclick=refreshJobs;

let scenes=[{prompt:'A wide establishing shot introduces the character and location, calm anticipation, cinematic natural sound',duration:5},{prompt:'A closer moving shot as the character begins the main action, building energy and visual detail',duration:5},{prompt:'A striking final shot resolves the moment and holds on an expressive ending, cinematic finish',duration:5}];
const sceneDurations=()=>selectedEngine()?.controls?.durations?.map(Number)||[5];
function renderScenes(){const root=$('#scenes'),durations=sceneDurations();root.innerHTML='';scenes.forEach((scene,i)=>{const el=document.createElement('div');el.className='scene';
  el.innerHTML='<div class="scene-head"><b>Scene '+(i+1)+'</b>'+(scenes.length>2?'<button class="danger" type="button">Remove</button>':'')+'</div><div class="scene-grid"><textarea placeholder="What happens in this shot?"></textarea><div><label>Length</label><select>'+durations.map(n=>'<option value="'+n+'" '+(Number(scene.duration)===n?'selected':'')+'>'+n+' sec</option>').join('')+'</select></div></div>';
  const ta=el.querySelector('textarea'),sel=el.querySelector('select');ta.value=scene.prompt;ta.oninput=()=>scene.prompt=ta.value;sel.onchange=()=>scene.duration=Number(sel.value);const rm=el.querySelector('.danger');if(rm)rm.onclick=()=>{scenes.splice(i,1);renderScenes();saveDraft();};const actions=document.createElement('div');actions.className='scene-actions';for(const [label,delta] of [['Move up',-1],['Move down',1],['Duplicate',0]]){const button=document.createElement('button');button.type='button';button.className='ghost';button.textContent=label;button.disabled=delta===0?scenes.length>=8:i+delta<0||i+delta>=scenes.length;button.onclick=()=>{if(delta===0)scenes.splice(i+1,0,{...scene});else [scenes[i],scenes[i+delta]]=[scenes[i+delta],scenes[i]];renderScenes();saveDraft();};actions.append(button);}el.querySelector('.scene-head').append(actions);root.appendChild(el);});}
$('#add-scene').onclick=()=>{if(scenes.length>=8)return;scenes.push({prompt:'',duration:sceneDurations()[0]});renderScenes();saveDraft();};renderScenes();

async function generateLab(){const engine=selectedEngine();try{if(!engine?.health?.ready)throw new Error('The selected model is offline. Your scene draft is saved.');if(scenes.some(x=>!x.prompt.trim()))throw new Error('Every scene needs a direction.');const payload={kind:'scenes',request_id:crypto.randomUUID(),story:$('#story').value.trim(),scenes,engine_id:engine.id,seed:$('#lab-seed').value.trim(),identity_lock:$('#lab-lock').checked,continuity:$('#continuity').checked,...settingsPayload()};if(imageData)payload.image_b64=imageData;$('#lab-go').disabled=true;const job=await api('/api/videos/jobs',payload);watchJob('scenes',job.id);$('#lab-status').textContent='Movie queued';saveDraft();await refreshJobs();}catch(e){$('#lab-status').textContent=e.message;}finally{$('#lab-go').disabled=false;}}
$('#lab-go').onclick=generateLab;

const draftFields=['p','prompt-format','s','duration','story','lab-seed','resolution','aspect-ratio','steps','guidance','negative-prompt'];
function saveDraft(){try{const fields=Object.fromEntries(draftFields.map(id=>[id,$('#'+id).value]));localStorage.setItem('frosty.video.draft',JSON.stringify({fields,engine:selectedEngineId,scenes,continuity:$('#continuity').checked}));}catch{}}
async function restoreDraft(){await refreshEngines();try{const saved=JSON.parse(localStorage.getItem('frosty.video.draft')||'null');if(!saved)return;if(engines.some(e=>e.id===saved.engine)){selectedEngineId=saved.engine;$('#engine-select').value=selectedEngineId;applyEngine();}for(const id of draftFields)if(typeof saved.fields?.[id]==='string')$('#'+id).value=saved.fields[id];promptMode=$('#prompt-format').value==='json'?'json':'text';promptDrafts[promptMode]=$('#p').value;$('#p').classList.toggle('json-prompt',promptMode==='json');$('#json-template').hidden=promptMode!=='json';$('#json-format').hidden=promptMode!=='json';updatePromptNote();if(Array.isArray(saved.scenes)&&saved.scenes.length>=2&&saved.scenes.length<=8&&saved.scenes.every(x=>typeof x.prompt==='string'&&Number.isFinite(x.duration)))scenes=saved.scenes;$('#continuity').checked=saved.continuity!==false;renderScenes();}catch{}}
document.addEventListener('input',saveDraft);document.addEventListener('change',saveDraft);
$('#apply-shot').onclick=()=>{const fields={camera:$('#shot-camera').value.trim(),lighting:$('#shot-light').value.trim(),motion:$('#shot-action').value.trim(),sound:$('#shot-audio').value.trim()};try{if(promptMode==='json'){const prompt=parsedJsonPrompt();prompt.shot_direction=Object.fromEntries(Object.entries(fields).filter(([,v])=>v));$('#p').value=JSON.stringify(prompt,null,2);}else{$('#p').value=[$('#p').value.trim(),...Object.entries(fields).filter(([,v])=>v).map(([k,v])=>k[0].toUpperCase()+k.slice(1)+': '+v)].filter(Boolean).join('\n');}promptDrafts[promptMode]=$('#p').value;updatePromptNote();saveDraft();}catch(e){$('#status').textContent=e.message;}};
$('#quality-preset').onchange=()=>{const c=selectedEngine()?.controls?.steps;if(!c)return;const scale={draft:.6,standard:1,high:1.5}[$('#quality-preset').value];$('#steps').value=Math.max(c.min,Math.min(c.max,Math.round(c.default*scale)));saveDraft();};
$('#choose-image-library').onclick=async()=>{const dialog=$('#image-library-dialog'),root=$('#image-library-items');dialog.showModal();root.textContent='Loading images…';try{const data=await api('/api/images/gallery');root.innerHTML='';for(const item of (data.items||[]).slice(0,100)){const button=document.createElement('button'),image=document.createElement('img'),label=document.createElement('span');image.src=item.file_url;image.alt=item.prompt||item.name;label.textContent=item.prompt||item.name;button.append(image,label);button.onclick=async()=>{try{const response=await fetch(item.file_url);if(!response.ok)throw new Error('Image is unavailable');const blob=await response.blob();await setImage(new File([blob],item.name,{type:blob.type}));dialog.close();}catch(e){root.textContent=e.message;}};root.append(button);}if(!root.children.length)root.textContent='No saved images yet. Generate one in Frosty Image first.';}catch(e){root.textContent=e.message;}};
$('#close-image-library').onclick=()=>$('#image-library-dialog').close();
restoreDraft();loadGallery();refreshJobs();setInterval(refreshEngines,10000);setInterval(refreshJobs,1800);
