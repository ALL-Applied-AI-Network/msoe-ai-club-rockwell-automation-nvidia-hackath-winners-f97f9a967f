"""Validate and atomically publish the completed v3_HZ checkpoint for local testing."""
import argparse,hashlib,json,os,shutil,tempfile
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE_SHA='453be3f451fe5e6f801ffd6b20e7096aa8cc1724244db13711c9ce4566c456af'


def gate(baseline,candidate):
    misses=lambda r:r['defects_accepted']+r['defects_as_empty']
    checks=dict(fewer_unflagged_defects=misses(candidate)<misses(baseline),
                higher_accuracy=candidate['accuracy']>baseline['accuracy'],
                no_extra_good_holds=candidate['good_held']<=baseline['good_held'])
    return dict(passed=all(checks.values()),checks=checks)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--job',type=Path,default=ROOT/'docs/validation/v3_HZ-training-job.json');a=ap.parse_args()
    job=json.loads(a.job.read_text());assert job['status']=='succeeded' and job['exit_code']==0
    source=a.source.resolve();target=ROOT/'model/v3_HZ'
    if target.exists():raise FileExistsError('v3_HZ already exists; published checkpoints are never overwritten')
    required=['model.pt','metrics.json','runtime.json','real-evaluation.json','v2-real-evaluation.json',
              'synthetic-test.json','v2-synthetic-test.json','data-manifest.json','input-provenance.json','camera-fit.json']
    assert all((source/f).is_file() for f in required),'incomplete training outputs'
    read=lambda f:json.loads((source/f).read_text())
    ck=torch.load(source/'model.pt',map_location='cpu',weights_only=True)
    assert ck['version']=='v3_HZ' and ck['crop_mode']=='review_bbox_v1' and ck['size']==128
    assert ck['classes']==['pass','missing_cap','crooked_cap','damaged_vial','no_vial']
    assert ck['training_source']=='synthetic-only'
    manifest=read('data-manifest.json')
    assert manifest['real_pixels_used_for_training'] is False
    assert manifest['counts']==dict(train=5760,val=1440,test=1440,real_eval=120)
    model_sha=hashlib.sha256((source/'model.pt').read_bytes()).hexdigest()
    assert hashlib.sha256((ROOT/'model/model.pt').read_bytes()).hexdigest()==BASE_SHA
    real,baseline=read('real-evaluation.json'),read('v2-real-evaluation.json')
    synthetic,old_synthetic=read('synthetic-test.json'),read('v2-synthetic-test.json')
    assert real['model_sha256']==synthetic['model_sha256']==model_sha
    assert baseline['model_sha256']==old_synthetic['model_sha256']==BASE_SHA
    assert real['all']['positions']==baseline['all']['positions']==120
    assert synthetic['all']['positions']==old_synthetic['all']['positions']==1440
    assert 'GB10' in read('runtime.json')['device']
    metrics=['positions','accuracy','macro_f1','defects','defects_accepted','defects_as_empty','good','good_held','uncertain']
    summary=lambda r:{k:r['all'][k] for k in metrics}
    result=dict(version='v3_HZ',source_job=job['job_id'],model_sha256=model_sha,
                released_for_testing=True,production_validated=False,
                training_source='synthetic-only',crop_mode='review_bbox_v1',
                development_gate=gate(baseline['all'],real['all']),
                real_reference=dict(scope=real['scope'],v2=summary(baseline),v3_HZ=summary(real)),
                synthetic_test=dict(v2=summary(old_synthetic),v3_HZ=summary(synthetic)),
                limitation='Only two reviewed real development reference images; independent real-line testing remains required.')
    temporary=Path(tempfile.mkdtemp(prefix='.v3_HZ-',dir=ROOT/'model'))
    try:
        shutil.copytree(source,temporary,dirs_exist_ok=True)
        (temporary/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        assert hashlib.sha256((temporary/'model.pt').read_bytes()).hexdigest()==model_sha
        assert temporary.resolve().parent==target.resolve().parent==(ROOT/'model').resolve()
        os.rename(temporary,target)
    finally:
        if temporary.exists():
            assert temporary.resolve().parent==(ROOT/'model').resolve() and temporary.name.startswith('.v3_HZ-')
            shutil.rmtree(temporary)
    (ROOT/'docs/validation/v3_HZ-final.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
