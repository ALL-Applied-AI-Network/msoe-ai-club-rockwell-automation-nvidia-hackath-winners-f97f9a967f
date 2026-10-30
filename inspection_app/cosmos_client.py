"""Reach the team's loopback-only GPU service with bounded SSH requests."""
import json,os,shlex,subprocess
import httpx


def call(path,payload=None,timeout=180):
    url=os.environ.get('COSMOS_URL')
    if url:
        with httpx.Client(timeout=timeout) as client:
            response=client.get(url.rstrip('/')+path) if payload is None else client.post(url.rstrip('/')+path,json=payload)
            response.raise_for_status();return response.json()
    if path not in ['/health','/reason']:raise ValueError('Unsupported Cosmos endpoint')
    script=("import sys,urllib.request;data=sys.stdin.buffer.read();"
            f"r=urllib.request.Request('http://127.0.0.1:8766{path}',data=data or None,headers={{'Content-Type':'application/json'}});"
            "print(urllib.request.urlopen(r,timeout=170).read().decode())")
    command='docker exec -i hz-cosmos python -c '+shlex.quote(script)
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',
                           os.environ.get('HZ_DGX_SSH','yt-winners7506@172.30.102.95'),command],
                          input=json.dumps(payload) if payload is not None else '',text=True,
                          encoding='utf-8',errors='replace',capture_output=True,timeout=timeout,check=False)
    if result.returncode:raise RuntimeError('DGX Cosmos request failed: '+result.stderr[-500:])
    return json.loads(result.stdout)
