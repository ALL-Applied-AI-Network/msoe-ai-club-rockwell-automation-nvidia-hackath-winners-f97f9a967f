// Exercise the actual browser export code without a browser download dependency.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');

class Element {
  constructor(tag='div'){this.tag=tag;this.value='';this.dataset={};this.style={};this.children=[];this.options=[];this.classList={add(){},remove(){}};this.checked=false;this.textContent='';}
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  addEventListener(){}
  setAttribute(){}
  querySelector(){return new Element('span');}
  remove(){}
  click(){if(this.tag==='a')downloads.push({name:this.download,blob:blobs.get(this.href)});}
}
const elements=new Map(),blobs=new Map(),downloads=[];
const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
get('model').value='v2';
const context=vm.createContext({
  document:{getElementById:get,createElement:tag=>new Element(tag),body:new Element('body'),addEventListener(){}},
  location:{pathname:'/'},crypto:{randomUUID:()=> 'test-session'},navigator:{},window:{addEventListener(){}},
  setInterval(){},setTimeout(){},clearTimeout(){},Blob,URL:{createObjectURL(blob){const id='blob:'+blobs.size;blobs.set(id,blob);return id;},revokeObjectURL(){}},
  fetch:async()=>({ok:true,json:async()=>({models:{v2:true,v3_HZ:false},device:'cpu',cosmos:{ready:false}})}),
  performance:{now:()=>0},console,Image:class{},File:class{},AbortController
});
const source=fs.readFileSync(path.join(__dirname,'../inspection_app/static/app.js'),'utf8');
vm.runInContext(source,context);
const example={model:'v2',model_sha256:'abc',overlay:'large annotated image',source_frame:'large raw image',
  slots:[{id:'G01',area:'good',class:'pass',confidence:.9,action:'accept'}]};

(async()=>{
  vm.runInContext(`result=${JSON.stringify(example)};exportJson();exportCsv();`,context);
  assert.equal(downloads.length,2);
  const json=JSON.parse(await downloads[0].blob.text());
  assert.equal(json.slots.length,1);assert.equal(json.overlay,undefined);assert.equal(json.source_frame,undefined);
  const csv=await downloads[1].blob.text();assert.match(csv,/'?G01/);assert.match(csv,/"confidence"/);assert.match(csv,/"0.9"/);
  vm.runInContext(`result.slots[0].id='=1+1';exportCsv();videoMeta={duration_seconds:3};videoFrames=[result,{...result,time_seconds:3}];exportJson();`,context);
  assert.match(await downloads[2].blob.text(),/"'=1\+1"/);
  const video=JSON.parse(await downloads[3].blob.text());assert.equal(video.frames.length,2);assert.equal(video.duration_seconds,3);
  assert(video.frames.every(frame=>frame.overlay===undefined && frame.source_frame===undefined));
  console.log('Frontend exports: JSON, CSV, formula guard, and sampled-video export passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
