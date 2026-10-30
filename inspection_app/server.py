"""Local camera application. Run: python -m uvicorn inspection_app.server:app --host 127.0.0.1 --port 8000"""
import base64,io,tempfile,threading,time,uuid,subprocess
from contextlib import asynccontextmanager
from pathlib import Path
import cv2,numpy as np,httpx
from PIL import Image,ImageOps,UnidentifiedImageError
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .engine import Engine,overlay,ROOT
from .sequence import TemporalTracker,summarize
from . import cosmos_client

APP=Path(__file__).resolve().parent
engine=Engine();sessions={};sessions_lock=threading.Lock()
MAX_IMAGE=15*1024*1024
cosmos_health={'ready':False,'message':'Checking DGX Cosmos service'}
cosmos_lock=threading.Lock()


def poll_cosmos(stop):
    global cosmos_health
    while not stop.is_set():
        try:
            health=cosmos_client.call('/health',timeout=12)
            if not isinstance(health,dict):raise ValueError('Invalid health response')
        except (httpx.HTTPError,ValueError,RuntimeError,subprocess.TimeoutExpired,OSError) as e:
            health={'ready':False,'message':'DGX Cosmos unavailable. Check CSI VPN, SSH access, and the GPU service.',
                    'error_type':type(e).__name__}
        with cosmos_lock:cosmos_health=health
        stop.wait(15)


@asynccontextmanager
async def lifespan(app):
    stop=threading.Event()
    threading.Thread(target=poll_cosmos,args=(stop,),daemon=True).start()
    yield
    stop.set()


app=FastAPI(title='Vial Inspection · HZ',docs_url='/api/docs',lifespan=lifespan)


def decode(data):
    if len(data)>MAX_IMAGE:raise ValueError('Image limit is 15 MB')
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width*im.height>12_000_000:raise ValueError('Image pixel limit exceeded')
            im=ImageOps.exif_transpose(im).convert('RGB')
            return cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError,OSError) as e:raise ValueError('Unreadable image') from e


def counts_from(good,scrap):
    result={}
    for key,value in [('good',good),('scrap',scrap)]:
        if value is None or str(value).strip()=='':continue
        value=str(value).strip()
        if not value.isascii() or not value.isdigit() or not 0<=int(value)<=30:
            raise ValueError('Placement counts must be whole numbers between 0 and 30')
        result[key]=int(value)
    return result


def perform(data,version,good,scrap,complete,session=None):
    counts=counts_from(good,scrap)
    image=decode(data);result=engine.inspect(image,version,counts,complete)
    if session:
        with sessions_lock:
            now=time.monotonic()
            for key in [k for k,v in sessions.items() if now-v[1]>900]:sessions.pop(key)
            key=(session,version)
            if key not in sessions and len(sessions)>=64:raise ValueError('Too many active sessions')
            tracker,_=sessions.setdefault(key,(TemporalTracker(),now))
            sessions[key]=(tracker,now);result['events']=tracker.update(result['slots'])
    ok,jpg=cv2.imencode('.jpg',overlay(image,result),[cv2.IMWRITE_JPEG_QUALITY,85])
    if not ok:raise ValueError('Cannot encode result')
    result['overlay']='data:image/jpeg;base64,'+base64.b64encode(jpg).decode()
    return result


@app.get('/api/status')
def status():
    with cosmos_lock:cosmos=dict(cosmos_health)
    return dict(device=engine.device,models={v:engine.model_path(v).exists() for v in ['v2','v3_HZ','v4_HZ']},cosmos=cosmos)


@app.post('/api/inspect')
def inspect(file:UploadFile=File(...),version:str=Form('v4_HZ'),good:str=Form(''),scrap:str=Form(''),
            complete:bool=Form(False),session:str=Form('')):
    try:
        if session:uuid.UUID(session)
        data=file.file.read(MAX_IMAGE+1)
        return perform(data,version,good,scrap,complete,session or None)
    except FileNotFoundError as e:raise HTTPException(503,str(e))
    except (ValueError,KeyError) as e:raise HTTPException(422,str(e))


@app.post('/api/video')
def video(file:UploadFile=File(...),version:str=Form('v4_HZ'),good:str=Form(''),scrap:str=Form(''),complete:bool=Form(False)):
    try:counts=counts_from(good,scrap)
    except ValueError as e:raise HTTPException(422,str(e))
    # Bound file size, duration sampled, and output size; video is deleted after inspection.
    with tempfile.TemporaryDirectory(prefix='hz-video-') as directory:
        path=Path(directory)/'input-video';total=0
        with path.open('wb') as out:
            while chunk:=file.file.read(1024*1024):
                total+=len(chunk)
                if total>200*1024*1024:raise HTTPException(413,'Video limit is 200 MB')
                out.write(chunk)
        capture=cv2.VideoCapture(str(path))
        if not capture.isOpened():capture.release();raise HTTPException(422,'Cannot decode this video')
        fps=capture.get(cv2.CAP_PROP_FPS);n=capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not np.isfinite(fps) or fps<=0 or not np.isfinite(n) or n<=0:
            capture.release();raise HTTPException(422,'Invalid video metadata')
        tracker=TemporalTracker();frames=[];final_image=None
        try:
            duration=n/fps;frame_count=max(1,int(round(n)))
            sample_count=min(30,frame_count,max(2,int(duration)+1))
            frame_indices=np.unique(np.rint(np.linspace(0,frame_count-1,sample_count)).astype(int))
            times=frame_indices/fps
            step=float(times[1]-times[0]) if len(times)>1 else duration
            for frame_index,sec in zip(frame_indices,times):
                capture.set(cv2.CAP_PROP_POS_FRAMES,int(frame_index));ok,image=capture.read()
                if not ok:continue
                result=engine.inspect(image,version,{},False)
                result['events']=tracker.update(result['slots']);result['time_seconds']=round(float(sec),2)
                encoded_ok,source=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,75])
                if not encoded_ok:raise ValueError('Cannot encode video source frame')
                result['source_frame']='data:image/jpeg;base64,'+base64.b64encode(source).decode()
                encoded_ok,encoded=cv2.imencode('.jpg',overlay(image,result),[cv2.IMWRITE_JPEG_QUALITY,65])
                if not encoded_ok:raise ValueError('Cannot encode video result')
                result['overlay']='data:image/jpeg;base64,'+base64.b64encode(encoded).decode();frames.append(result)
                final_image=image
        except FileNotFoundError as e:raise HTTPException(503,str(e))
        except ValueError as e:raise HTTPException(422,str(e))
        finally:capture.release()
        if not frames:raise HTTPException(422,'No decodable frames')
        # A single video-level count/completion signal belongs to the final sampled frame.
        final=frames[-1]
        final['groups']=summarize(final['slots'],counts,complete)
        ok,encoded=cv2.imencode('.jpg',overlay(final_image,final),[cv2.IMWRITE_JPEG_QUALITY,65])
        if not ok:raise HTTPException(422,'Cannot encode final result')
        final['overlay']='data:image/jpeg;base64,'+base64.b64encode(encoded).decode()
        return dict(frames=frames,duration_seconds=duration,sampling_interval_seconds=step,
                    note='At most 30 sampled frames; placements between samples can be missed. Placement counts and the good-order completion signal apply only to the final sampled frame.')


@app.post('/api/reason')
def reason(file:UploadFile=File(...),question:str=Form('Inspect visible vial conditions and explain uncertainties.')):
    try:
        image=decode(file.file.read(MAX_IMAGE+1));ok,encoded=cv2.imencode('.jpg',image)
        if not ok:raise ValueError('Cannot encode image for Cosmos')
        payload=dict(images=[base64.b64encode(encoded).decode()],question=question[:2000])
        return cosmos_client.call('/reason',payload,timeout=180)
    except ValueError as e:raise HTTPException(422,str(e))
    except (httpx.HTTPError,RuntimeError,subprocess.TimeoutExpired,OSError):
        raise HTTPException(503,'Cannot reach DGX Cosmos or the service is busy. Check CSI VPN, SSH, and GPU service readiness.')


@app.get('/api/examples/{name}')
def example(name:str):
    if name not in ['tray_000001.jpg','tray_000122.jpg']:raise HTTPException(404)
    return FileResponse(APP/'references'/name)


app.mount('/assets',StaticFiles(directory=APP/'static'),name='assets')


@app.get('/')
@app.get('/live')
def home():return FileResponse(APP/'static/index.html')
