'use strict';
const $ = id => document.getElementById(id);
const livePage = location.pathname === '/live';
const labels = {pass:'Pass',missing_cap:'Missing cap',crooked_cap:'Crooked cap',damaged_vial:'Damaged vial',no_vial:'Empty',uncertain:'Review'};
const shortLabels = {pass:'PASS',missing_cap:'NO CAP',crooked_cap:'TILTED',damaged_vial:'DAMAGE',no_vial:'EMPTY',uncertain:'REVIEW'};
const colors = {pass:'#93cc70',missing_cap:'#ffc568',crooked_cap:'#f4a4d7',damaged_vial:'#ff8a75',no_vial:'#9fc4db',uncertain:'#e0d7ac'};
let currentFile = null, rawImage = null, sourceUrl = '', result = null, selected = null;
let camera = null, monitoring = false, requestBusy = false, reasonBusy = false, backend = null;
let generation = 0, controller = null, monitorTimer = null, session = crypto.randomUUID();
let videoFrames = [], videoMeta = null, events = [], lastResultAt = 0, videoFrameGeneration = 0;

function text(id, value) { $(id).textContent = value; }
function error(message) { text('error', message); $('error').hidden = !message; }
function notice(message) { text('notice', message); $('notice').hidden = !message; }
function modelNotice(message) {text('model-warning',message);$('model-warning').hidden=!message;}
function modelAvailable() { return Boolean(backend?.models?.[$('model').value]); }
function setBusy(value, message='Inspecting 60 positions…') {
  requestBusy=value; $('busy').hidden=!value; text('busy-text',message); refreshControls();
}
function refreshControls() {
  const available=modelAvailable();
  $('inspect').disabled=requestBusy || !available || (!currentFile && !camera);
  $('inspect').querySelector('span').textContent=camera && !livePage ? 'Capture & inspect' : 'Run inspection';
  $('start-monitor').disabled=!camera || !available || requestBusy || monitoring;
  $('start-camera').disabled=Boolean(camera) || requestBusy;
  $('stop-camera').disabled=!camera && !monitoring;
  $('snapshot-camera').disabled=requestBusy;
  $('reason').disabled=reasonBusy || !backend?.cosmos?.ready || (!currentFile && !camera && !rawImage);
  $('export-json').disabled=!result; $('export-csv').disabled=!result;
  $('clear').disabled=!currentFile && !camera && !result;
  $('toggle-overlay').disabled=!result;
  $('model').disabled=requestBusy || monitoring;
  const version=$('model').value;
  text('model-state',backend ? (available ? (version==='v2'?'GitHub model/model.pt · diagnostic':'Checkpoint available') : 'Checkpoint unavailable / training') : 'Checking checkpoint…');
  if (version==='v2') modelNotice('Cross-camera diagnostic: v2 uses the GitHub model/model.pt checkpoint, built for different crops and a 640 × 480 camera view. Its predictions on this 60-position view require validation.');
  else if (backend && !available) modelNotice('v3_HZ is not available yet. The interface will detect its checkpoint when training completes. No inspection results are simulated.');
  else modelNotice('');
}
async function checkStatus() {
  try {
    const response=await fetch('/api/status'); if (!response.ok) throw new Error('Backend unavailable');
    backend=await response.json();
    text('backend-status',`Model server · ${backend.device.toUpperCase()}`); $('backend-dot').classList.add('ready');
    text('cosmos-status',backend.cosmos?.ready ? `${backend.cosmos.device || 'DGX Spark'} · ready` : (backend.cosmos?.error ? 'DGX service needs attention' : 'DGX service not ready'));
  } catch (_) {
    backend=null; text('backend-status','Model server unavailable'); $('backend-dot').classList.remove('ready'); text('cosmos-status','Model server unavailable');
  }
  refreshControls();
}
function placementFields(form) {
  for (const [name,id] of [['good','good-count'],['scrap','scrap-count']]) {
    const value=$(id).value.trim();
    if (value!=='' && (!/^\d+$/.test(value) || Number(value)>30)) throw new Error('Placement counts must be whole numbers from 0 to 30.');
    form.append(name,value);
  }
  form.append('complete',String($('order-complete').checked)); form.append('version',$('model').value);
}
async function api(path,form,signal) {
  const response=await fetch(path,{method:'POST',body:form,signal});
  let body; try { body=await response.json(); } catch (_) { throw new Error(`Server returned ${response.status}.`); }
  if (!response.ok) throw new Error(typeof body.detail==='string' ? body.detail : `Inspection failed (${response.status}).`);
  return body;
}
function revokeSource() { if (sourceUrl.startsWith('blob:')) URL.revokeObjectURL(sourceUrl); sourceUrl=''; }
async function imageFrom(url) {
  return new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>resolve(image);image.onerror=()=>reject(new Error('Cannot display this image.'));image.src=url;});
}
function clearResults() {
  result=null; selected=null; videoFrames=[]; videoMeta=null; events=[];
  $('timeline').hidden=true; $('live-overlay').hidden=true; $('stage-label').hidden=true;
  text('defect-count','—'); text('review-count','—'); text('inference-time','No inference yet');
  text('alignment-state','Camera alignment is checked before inference.');
  $('tray-summary').replaceChildren(element('p','small-text','Results appear after inspection.'));
  $('cosmos-answer').hidden=true; $('cosmos-answer').replaceChildren(); renderMaps(); renderDetail(); renderEvents(); refreshControls();
  $('toggle-overlay').dataset.hidden='false';text('toggle-overlay','Hide labels');
}
function stopCamera() {
  monitoring=false; clearTimeout(monitorTimer); monitorTimer=null; generation++;
  controller?.abort(); controller=null;
  if (camera) camera.getTracks().forEach(track=>track.stop()); camera=null;
  $('camera-video').srcObject=null; $('camera-video').hidden=true; $('live-overlay').hidden=true;
  $('source-dot').classList.remove('connected'); requestBusy=false; $('busy').hidden=true;
  if (!currentFile) { $('empty-state').hidden=false; $('stage-label').hidden=true; }
  text('start-monitor','Start monitoring'); refreshControls();
}
async function setFile(file) {
  if (file.size>15*1024*1024) throw new Error('Image limit is 15 MB.');
  const url=URL.createObjectURL(file);
  let image; try { image=await imageFrom(url); } catch (e) {URL.revokeObjectURL(url);throw e;}
  if (image.naturalWidth!==1920 || image.naturalHeight!==1080) {
    URL.revokeObjectURL(url); throw new Error(`This image is ${image.naturalWidth} × ${image.naturalHeight}. Inspection requires the calibrated 1920 × 1080 view. Do not resize a different camera view to bypass calibration.`);
  }
  stopCamera(); error(''); clearResults();
  revokeSource(); sourceUrl=url; currentFile=file; rawImage=image;
  $('frame-image').src=url; $('frame-image').hidden=false; $('empty-state').hidden=true;
  text('frame-name',file.name || 'Camera snapshot'); text('source-title','Inspection frame');
  text('source-meta','1920 × 1080 calibrated view'); text('stage-label-text','Source frame'); $('stage-label').hidden=false;
  refreshControls();
}
async function inspectCurrent() {
  if (requestBusy) return;
  let current;
  try {
    if (camera && !livePage) {
      const blob=await capture(); await setFile(new File([blob],`camera-${new Date().toISOString().replaceAll(':','-')}.jpg`,{type:'image/jpeg'}));
    }
    if (!currentFile) throw new Error('Choose an image or capture a camera frame first.');
    error(''); const form=new FormData();placementFields(form);form.append('file',currentFile);
    current=++generation;controller=new AbortController();setBusy(true);
    const data=await api('/api/inspect',form,controller.signal);
    if (current!==generation) return;
    renderResult(data); $('frame-image').src=data.overlay; text('stage-label-text',`${data.model} · labeled frame`);
  } catch (e) { if(e.name!=='AbortError') error(e.message); }
  finally {if(current===undefined || current===generation)setBusy(false);}
}
async function capture() {
  const video=$('camera-video');
  if (!camera || video.readyState<2) throw new Error('Camera is not ready.');
  if (video.videoWidth!==1920 || video.videoHeight!==1080) throw new Error(`Camera provided ${video.videoWidth} × ${video.videoHeight}. Choose a camera that supports calibrated 1920 × 1080 capture.`);
  const canvas=$('capture-canvas');canvas.width=1920;canvas.height=1080;canvas.getContext('2d').drawImage(video,0,0);
  return new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(new Error('Camera capture failed.')),'image/jpeg',.92));
}
async function connectCamera() {
  try {
    error(''); stopCamera(); clearResults(); currentFile=null;rawImage=null; revokeSource();
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires a supported browser at localhost or HTTPS.');
    const device=$('camera-device').value;
    camera=await navigator.mediaDevices.getUserMedia({audio:false,video:{width:{exact:1920},height:{exact:1080},...(device?{deviceId:{exact:device}}:{})}});
    const video=$('camera-video');video.srcObject=camera;video.hidden=false;
    await video.play();
    if (video.videoWidth!==1920 || video.videoHeight!==1080) {const dimensions=`${video.videoWidth} × ${video.videoHeight}`;stopCamera();throw new Error(`Camera returned ${dimensions}; calibrated 1920 × 1080 is required.`);}
    $('frame-image').hidden=true;$('empty-state').hidden=true;$('stage-label').hidden=false;
    text('stage-label-text','Camera connected · inference stopped');text('frame-name','Camera · 1920 × 1080');text('source-title',livePage?'Live camera':'Camera snapshot');
    $('source-dot').classList.add('connected');
    await refreshCameraDevices(camera.getVideoTracks()[0].getSettings().deviceId);session=crypto.randomUUID();refreshControls();
  } catch (e) {
    if (camera) stopCamera();
    const explanations={NotAllowedError:'Camera permission was denied. Allow camera access in your browser and try again.',NotFoundError:'No camera was found. Connect a camera or upload a video.',OverconstrainedError:'This camera cannot provide the required 1920 × 1080 view. Choose another camera; frames are never blindly resized.',NotReadableError:'The camera is in use or cannot be opened. Close other camera apps and retry.'};
    error(explanations[e.name] || e.message);
    await refreshCameraDevices();
  }
}
async function refreshCameraDevices(selectedDevice) {
  if(!navigator.mediaDevices?.enumerateDevices)return;
  try {
    const current=selectedDevice || $('camera-device').value;
    const devices=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput' && d.deviceId);
    $('camera-device').replaceChildren(new Option('Default camera',''),...devices.map((d,i)=>new Option(d.label || `Camera ${i+1}`,d.deviceId)));
    if([...$('camera-device').options].some(option=>option.value===current))$('camera-device').value=current;
  }catch(_){/* Device labels can remain unavailable until the user grants camera access. */}
}
async function monitor() {
  if (!camera || monitoring || !modelAvailable()) return;
  monitoring=true;session=crypto.randomUUID();const current=++generation;events=[];renderEvents();refreshControls();
  text('start-monitor','Monitoring');text('stage-label-text','Sampled inference · starting');
  async function tick() {
    if (!monitoring || current!==generation) return;
    const started=performance.now();
    try {
      const file=await capture();const form=new FormData();placementFields(form);form.append('file',file,'camera.jpg');form.append('session',session);
      controller=new AbortController();requestBusy=true;
      const data=await api('/api/inspect',form,controller.signal);
      if (!monitoring || current!==generation) return;
      renderResult(data);lastResultAt=performance.now();drawLive(data);
      const interval=Math.max(1000,performance.now()-started);
      text('stage-label-text',`Sampled inference · ${(1000/interval).toFixed(2)}/s`);
      text('inference-time',`${data.elapsed_ms.toFixed(0)} ms inference · ${(1000/interval).toFixed(2)} samples/s`);
    } catch (e) {
      if (e.name!=='AbortError') {error(e.message);$('live-overlay').hidden=true;monitoring=false;text('stage-label-text','Inference paused · inspect alignment');text('start-monitor','Resume monitoring');}
    } finally {
      requestBusy=false;refreshControls();
      if (monitoring && current===generation) monitorTimer=setTimeout(tick,Math.max(0,1000-(performance.now()-started)));
    }
  }
  await tick();
}
function drawLive(data) {
  const canvas=$('live-overlay');canvas.width=1920;canvas.height=1080;const context=canvas.getContext('2d');context.clearRect(0,0,1920,1080);
  if ($('toggle-overlay').dataset.hidden==='true') {canvas.hidden=true;return;}
  for (const slot of data.slots) {
    const [x,y]=slot.center;const color=slot.action==='pending'?'#b6bfb4':colors[slot.class];
    context.strokeStyle=color;context.lineWidth=4;context.beginPath();context.arc(x,y,18,0,Math.PI*2);context.stroke();
    context.font='17px monospace';const tag=`${slot.id} ${shortLabels[slot.class]}`;const width=context.measureText(tag).width;
    context.fillStyle='#17231cd9';context.fillRect(x-20,y-46,width+10,23);context.fillStyle=color;context.fillText(tag,x-15,y-29);
  }
  canvas.hidden=false;
}
function element(tag,className,content) { const node=document.createElement(tag);if(className)node.className=className;if(content!==undefined)node.textContent=content;return node; }
function renderResult(data) {
  result=data;lastResultAt=performance.now();
  const defects=data.slots.filter(s=>s.action==='defect').length;
  text('defect-count',String(defects));text('review-count',String(data.slots.filter(s=>s.class==='uncertain').length));
  const tray=$('tray-summary');tray.replaceChildren();
  for(const area of ['good','scrap']) {
    const group=data.groups[area];const block=element('div');const row=element('div','tray-row');row.append(element('span','',`${area==='good'?'G · Good':'S · Scrap'} tray`),element('strong','',`${group.occupied}/30 occupied`));
    const progress=element('div','progress'),bar=element('i');bar.style.width=`${group.occupied/30*100}%`;progress.append(bar);block.append(row,progress);
    const flags=[];if(group.missing.length)flags.push(`${group.missing.length} confirmed missing`);if(group.possible_gaps.length)flags.push(`${group.possible_gaps.length} possible gaps`);if(group.pending.length)flags.push(`${group.pending.length} pending`);
    if(flags.length)block.append(element('div','tray-flags',flags.join(' · ')));tray.append(block);
  }
  text('alignment-state',`${data.alignment.profile} · offset ${data.alignment.offset.join(', ')} px · residual ${data.alignment.residual.toFixed(3)}. Check rings visually.`);
  text('inference-time',`${data.elapsed_ms.toFixed(0)} ms · ${data.device.toUpperCase()} · ${data.model}`);
  events=[...(data.events || []).map(event=>({...event,time:typeof data.time_seconds==='number'?`${data.time_seconds.toFixed(1)} s`:new Date().toLocaleTimeString()})),...events].slice(0,60);
  renderMaps();renderDetail();renderEvents();refreshControls();
}
function renderMaps() {
  for(const [prefix,id] of [['G','good-grid'],['S','scrap-grid']]) {
    const container=$(id);container.replaceChildren();
    for(let i=1;i<=30;i++) {
      const key=prefix+String(i).padStart(2,'0'),slot=result?.slots.find(s=>s.id===key);
      const tile=element('button',`slot${slot?` class-${slot.class}`:''}${selected===key?' selected':''}`);tile.type='button';tile.style.gridColumn=String(Math.floor((i-1)/5)+1);tile.style.gridRow=String((i-1)%5+1);
      tile.append(element('strong','',key),element('small','',slot?(slot.action==='pending'?'PENDING':shortLabels[slot.class]):'—'));
      tile.setAttribute('aria-label',`${key}: ${slot?labels[slot.class]+', '+slot.action:'no result'}`);tile.setAttribute('aria-pressed',String(selected===key));
      tile.addEventListener('click',()=>{selected=key;renderMaps();renderDetail();});container.append(tile);
    }
  }
  for(const area of ['good','scrap'])text(`${area}-map-state`,result?`${result.groups[area].occupied} occupied / 30`:'30 positions');
}
function renderDetail() {
  const detail=$('slot-detail');detail.replaceChildren(element('span','tiny-label','POSITION DETAILS'));
  const slot=result?.slots.find(s=>s.id===selected);
  if(!slot){detail.append(element('h3','',selected || 'Select a position'),element('p','',selected?'Run inspection to see the result for this position.':'Choose a tile to inspect its class, confidence, and placement state.'));return;}
  detail.append(element('h3','',`${slot.id} · ${slot.area==='good'?'Good':'Scrap'} tray`),element('span','detail-class',labels[slot.class]));
  for(const [name,value] of [['Confidence',`${(slot.confidence*100).toFixed(1)}%`],['Placement state',slot.action.replaceAll('_',' ')],['Sequence',`Column ${Math.floor((slot.position-1)/5)+1}, row ${(slot.position-1)%5+1}`]]){const row=element('div','detail-stat');row.append(element('span','',name),element('strong','',value));detail.append(row);}
  if(slot.area==='scrap')detail.append(element('p','','Appearance alone cannot establish whether an upstream routing decision was wrong.'));
  if(rawImage){const canvas=document.createElement('canvas');const [l,t,r,b]=slot.bbox;canvas.width=r-l;canvas.height=b-t;canvas.getContext('2d').drawImage(rawImage,l,t,r-l,b-t,0,0,r-l,b-t);const image=element('img','detail-crop');image.alt=`Raw camera crop at ${slot.id}`;image.src=canvas.toDataURL('image/jpeg');detail.append(image);}
}
function renderEvents() {
  $('events-section').hidden=!livePage;const container=$('events');container.replaceChildren();
  if(!events.length){container.append(element('p','small-text','No stable changes observed yet.'));return;}
  for(const event of events){const row=element('div','event-row');row.append(element('span','',event.time),element('strong','',event.position),element('span','',`${event.event==='placement_observed'?'Placement observed':'Appearance changed'} · ${labels[event.previous]} → ${labels[event.current]}`));container.append(row);}
}
async function loadExample(name) {
  try {error('');const response=await fetch('/api/examples/'+name);if(!response.ok)throw new Error('Example image is unavailable.');await setFile(new File([await response.blob()],name,{type:'image/jpeg'}));}catch(e){error(e.message);}
}
async function uploadVideo(file) {
  if(!file)return;
  try {
    stopCamera();clearResults();currentFile=null;rawImage=null;revokeSource();error('');
    if(file.size>200*1024*1024)throw new Error('Video limit is 200 MB.');
    const form=new FormData();placementFields(form);form.append('file',file);
    const current=++generation;controller=new AbortController();setBusy(true,'Inspecting sampled video frames…');
    const data=await api('/api/video',form,controller.signal);if(current!==generation)return;
    videoFrames=data.frames;videoMeta={duration_seconds:data.duration_seconds,sampling_interval_seconds:data.sampling_interval_seconds,note:data.note};
    $('timeline-range').max=String(videoFrames.length-1);$('timeline-range').value='0';$('timeline').hidden=false;
    $('frame-image').hidden=false;$('empty-state').hidden=true;$('stage-label').hidden=false;
    text('frame-name',file.name);text('source-title','Sampled video');notice(data.note);await showVideoFrame(0);
  }catch(e){if(e.name!=='AbortError')error(e.message);}finally{setBusy(false);}
}
async function showVideoFrame(index) {
  const frame=videoFrames[index];if(!frame)return;
  const current=++videoFrameGeneration;
  const image=frame.source_frame?await imageFrom(frame.source_frame):null;
  const file=frame.source_frame?new File([await (await fetch(frame.source_frame)).blob()],`video-${frame.time_seconds}.jpg`,{type:'image/jpeg'}):null;
  if(current!==videoFrameGeneration)return;
  $('frame-image').src=$('toggle-overlay').dataset.hidden==='true'&&image?image.src:frame.overlay;rawImage=image;currentFile=file;
  events=[];for(const data of videoFrames.slice(0,index))events.push(...(data.events||[]).map(e=>({...e,time:`${data.time_seconds.toFixed(1)} s`})));
  events.reverse();renderResult(frame);text('timeline-time',`${frame.time_seconds.toFixed(1)} s`);text('stage-label-text',`Sample ${index+1}/${videoFrames.length} · ${frame.time_seconds.toFixed(1)} s`);
}
async function askCosmos() {
  if(reasonBusy)return;
  try {
    error('');reasonBusy=true;refreshControls();text('reason','Reasoning on DGX…');
    let blob=camera?await capture():currentFile;if(!blob)throw new Error('Choose an image, camera frame, or sampled video frame first.');
    let question=$('cosmos-question').value.trim();const slot=result?.slots.find(s=>s.id===selected);
    if(slot && !camera && rawImage){const [l,t,r,b]=slot.bbox;const canvas=document.createElement('canvas');canvas.width=r-l;canvas.height=b-t;canvas.getContext('2d').drawImage(rawImage,l,t,r-l,b-t,0,0,r-l,b-t);blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',.95));question=`Target position ${slot.id}; this context crop may contain neighboring vials. ${question}`;}
    const form=new FormData();form.append('file',blob,'inspection.jpg');form.append('question',question);
    $('cosmos-answer').hidden=false;$('cosmos-answer').replaceChildren(element('span','','Cosmos is inspecting the visible evidence…'));
    const data=await api('/api/reason',form);$('cosmos-answer').replaceChildren(element('div','',data.answer),element('small','',`${data.model} · ${data.elapsed_seconds}s · advisory`));
  }catch(e){$('cosmos-answer').hidden=false;$('cosmos-answer').replaceChildren(element('span','',e.message));}
  finally{reasonBusy=false;$('reason').replaceChildren(element('span','','Ask Cosmos'),element('span','','↗'));refreshControls();}
}
function cleanResult(value){const {overlay,source_frame,...rest}=value;return rest;}
function download(name,content,type){const url=URL.createObjectURL(new Blob([content],{type}));const a=element('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function exportJson(){if(!result)return;const data=videoFrames.length?{...videoMeta,frames:videoFrames.map(cleanResult)}:cleanResult(result);download(`vial-inspection-${result.model}.json`,JSON.stringify(data,null,2),'application/json');}
function exportCsv(){if(!result)return;const values=[['sample_time_seconds','position','tray','class','confidence','placement_action','stable','model','checkpoint_sha256']];for(const frame of (videoFrames.length?videoFrames:[result]))for(const s of frame.slots)values.push([frame.time_seconds??'',s.id,s.area,s.class,s.confidence,s.action,s.stable??'',frame.model,frame.model_sha256]);const escape=value=>{let v=String(value);if(/^[=+\-@\t\r]/.test(v))v="'"+v;return '"'+v.replaceAll('"','""')+'"';};download(`vial-inspection-${result.model}.csv`,values.map(row=>row.map(escape).join(',')).join('\r\n'),'text/csv');}

$('choose-image').onclick=()=>$('image-input').click();$('image-input').onchange=async event=>{const file=event.target.files[0];event.target.value='';if(file)try{await setFile(file);}catch(e){error(e.message);}};
$('snapshot-camera').onclick=connectCamera;$('inspect').onclick=inspectCurrent;$('start-camera').onclick=connectCamera;$('start-monitor').onclick=monitor;$('stop-camera').onclick=stopCamera;
$('camera-device').onchange=()=>{if(camera)connectCamera();};$('model').onchange=()=>{if(monitoring)stopCamera();clearResults();if(rawImage)$('frame-image').src=rawImage.src;refreshControls();};
$('example-1').onclick=()=>loadExample('tray_000001.jpg');$('example-122').onclick=()=>loadExample('tray_000122.jpg');
$('choose-video').onclick=()=>{if(!modelAvailable()){error('The selected model checkpoint is not available yet.');return;}$('video-input').click();};$('video-input').onchange=event=>{const file=event.target.files[0];event.target.value='';uploadVideo(file);};
$('timeline-range').oninput=event=>showVideoFrame(Number(event.target.value));$('reason').onclick=askCosmos;$('export-json').onclick=exportJson;$('export-csv').onclick=exportCsv;
$('toggle-overlay').onclick=()=>{const hidden=$('toggle-overlay').dataset.hidden!=='true';$('toggle-overlay').dataset.hidden=String(hidden);text('toggle-overlay',hidden?'Show labels':'Hide labels');if(camera){if(hidden)$('live-overlay').hidden=true;else if(result)drawLive(result);}else if(result)$('frame-image').src=hidden&&rawImage?rawImage.src:result.overlay;};
$('clear').onclick=()=>{stopCamera();currentFile=null;rawImage=null;revokeSource();clearResults();$('frame-image').hidden=true;$('empty-state').hidden=false;text('frame-name','No image selected');error('');};
for(const event of ['dragenter','dragover'])$('drop-target').addEventListener(event,e=>{e.preventDefault();if(!livePage)$('drop-target').classList.add('dragover');});
for(const event of ['dragleave','drop'])$('drop-target').addEventListener(event,e=>{e.preventDefault();$('drop-target').classList.remove('dragover');});
$('drop-target').addEventListener('drop',async e=>{if(livePage)return;const file=e.dataTransfer.files[0];if(file)try{await setFile(file);}catch(errorValue){error(errorValue.message);}});
window.addEventListener('beforeunload',()=>{camera?.getTracks().forEach(track=>track.stop());controller?.abort();revokeSource();});
document.addEventListener('visibilitychange',()=>{if(document.hidden && monitoring){stopCamera();notice('Monitoring stopped while this tab was hidden. Reconnect the camera to resume.');}});
if(livePage){$('nav-image').classList.remove('active');$('nav-live').classList.add('active');$('image-tools').hidden=true;$('live-tools').hidden=false;text('page-title','Live monitor');text('page-description','Observe the fill pattern as the robot places each vial.');text('empty-title','Connect the station camera');text('empty-description','Connect a calibrated camera, then start sampled inspection. Or upload a recorded station video.');text('empty-foot','CAMERA 1920 × 1080 · VIDEO UP TO 200 MB');text('count-description','For live capture, counts apply to the current frame. For uploaded video, signals apply to the final sampled frame.');}
renderMaps();renderEvents();checkStatus();refreshCameraDevices();navigator.mediaDevices?.addEventListener('devicechange',()=>refreshCameraDevices());setInterval(checkStatus,8000);
setInterval(()=>{if(camera && result && performance.now()-lastResultAt>4000){$('live-overlay').hidden=true;if(monitoring)text('stage-label-text','Waiting for next inference · previous labels expired');}},1000);
