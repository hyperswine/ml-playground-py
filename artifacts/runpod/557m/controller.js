// Executed in functions.exec. Uses managed MCP authorization; no API key is
// read, copied to a pod, or stored in this file. Resource IDs are this run's only.
// This requires an active client connection. Exec cells can expire across
// sessions, so this is not a durable unattended budget scheduler.
const state = load('qos557_controller');
if (!state || !state.pod_id || !state.created_unix) throw Error('Missing owned run state');
const base = '/Users/jasonqin/Documents/GitHub/Playground/python-playground-UV';
const evidence = base + '/qos_ai_benchmark/evidence/runpod_557m_20261019';
function unpack(result) {
  if (result.isError) throw Error(JSON.stringify(result.content));
  for (const c of result.content || []) if (c.type === 'text') {
    try { return JSON.parse(c.text); } catch {}
  }
  throw Error('Missing structured response');
}
function network(p) {
  p = p.pod || p;
  const runtime = p.runtime || {};
  const direct = p.ssh?.direct || {};
  const ports = runtime.ports || [];
  const entry = Array.isArray(ports) ? ports.find(x => x.privatePort === 22 || x.port === 22) : null;
  const ip = direct.host || p.publicIp || p.publicIpAddress || runtime.publicIp || entry?.ip;
  const port = direct.port || p.portMappings?.['22'] || entry?.publicPort;
  if (!/^\d{1,3}(\.\d{1,3}){3}$/.test(ip || '') || !Number.isInteger(Number(port)) || Number(port) < 1 || Number(port) > 65535) return null;
  return {ip, port: Number(port)};
}
function sshPrefix(n) {
  return `ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519 -i ~/.ssh/id_ed25519_hyperswine -p ${n.port} root@${n.ip}`;
}
async function shell(cmd, label, max = 1000) {
  return await tools.exec_command({cmd, sandbox_permissions:'require_escalated', justification:label,
    yield_time_ms:1000, max_output_tokens:max});
}
async function finishCommand(r, deadline) {
  let output = r.output || '';
  while (r.session_id) {
    if (Date.now()/1000 > deadline) throw Error('Transfer exceeded its reserved deadline');
    r = await tools.write_stdin({session_id:r.session_id, chars:'', yield_time_ms:1000, max_output_tokens:1000});
    output += r.output || '';
  }
  if (r.exit_code !== 0) throw Error('Command failed: ' + output.slice(-2000));
  return output;
}
async function stopOwned(id) {
  let p = unpack(await tools.mcp__runpod__get_pod({id}));
  const body = p.pod || p;
  if (body.desiredStatus !== 'EXITED' && body.status !== 'EXITED') {
    unpack(await tools.mcp__runpod__pod_action({id, body:{action:'stop'}}));
  }
  for(let attempt=0;attempt<12;attempt++) {
    p = unpack(await tools.mcp__runpod__get_pod({id}));
    const q = p.pod || p;
    if (q.desiredStatus === 'EXITED' || q.status === 'EXITED') return p;
    await new Promise(r=>setTimeout(r,5000));
  }
  throw Error('GPU stop read-back was not confirmed');
}
const hard = state.created_unix + 8*3600;
let terminal = state.verified_terminal || null;
let lastPhase = null;
let sshFailures = 0;
store('qos557_watchdog_armed', {pod_id:state.pod_id, hard_stop_unix:hard});
notify({watchdog:'armed',pod_id:state.pod_id,hard_stop_unix:hard});
await yield_control();
try {
  while(!terminal && Date.now()/1000 < hard) {
    const p = unpack(await tools.mcp__runpod__get_pod({id:state.pod_id}));
    if ((p.pod||p).status === 'EXITED') {terminal={phase:'externally_stopped'};break;}
    const n = network(p);
    if (n) {
      try {
        let r = await shell(sshPrefix(n) + ` 'cat /workspace/qos_ai_557m/experiment/status.json 2>/dev/null || true'`,
          'Read the owned training job status for budget supervision');
        const out = await finishCommand(r, Math.min(hard, Date.now()/1000+45));
        sshFailures=0;
        const jsonStart = out.indexOf('{');
        if(jsonStart >= 0) {
          const s = JSON.parse(out.slice(jsonStart).trim()); store('qos557_live_status',s);
          if (s.phase !== lastPhase) {notify(s);lastPhase=s.phase;}
          if (s.phase === 'complete' || s.phase === 'failed') {terminal=s;break;}
        }
      } catch(e) {
        sshFailures++;
        notify({supervisor_ssh_failures:sshFailures,error:String(e)});
        if(sshFailures>=5)throw e;
      }
    }
    if (lastPhase === null && Date.now()/1000 > state.created_unix+1800) {
      terminal={phase:'setup_deadline',reason:'No readable job status within setup allowance'};break;
    }
    await new Promise(r=>setTimeout(r,30000));
  }
  terminal ||= {phase:'hard_deadline',unix_time:Date.now()/1000};
  store('qos557_terminal_status',terminal);
  const stopped = await stopOwned(state.pod_id);
  store('qos557_gpu_stopped',stopped);
  notify({gpu_stopped:true,pod_id:state.pod_id,terminal});
  if (terminal.phase !== 'complete') {
    store('qos557_controller_needs_attention',{terminal,volume_id:state.volume_id});
    notify({phase:'needs_attention',gpu_stopped:true,preserved_volume_id:state.volume_id});
    return;
  }
  // Backup on a small CPU pod, so file transfer never burns GPU credit.
  // Fresh catalog reads precede the new billable create. Cap CPU at US$0.08/h.
  let backupId = state.backup_pod_id || null;
  if (backupId) {
    const existing = unpack(await tools.mcp__runpod__get_pod({id:backupId}));
    const q = existing.pod || existing;
    if (q.status !== 'RUNNING' || !q.mounts?.network?.some(m=>m.volumeId===state.volume_id && m.path==='/workspace')) {
      throw Error('Existing backup pod must be running with this run\'s verified network mount');
    }
    notify({phase:'backup_cpu_reused',pod_id:backupId,volume_id:state.volume_id});
  } else {
  const catalog = unpack(await tools.mcp__runpod__list_cpu_types({include:['AVAILABILITY'],product:['POD'],vcpuCount:2}));
  const cpu = catalog.cpus.filter(c=>c.price?.securePerVcpu*2 <= .08 && c.dataCenters?.some(d=>d.id==='EU-RO-1' && d.availability!=='NONE'))
    .sort((a,b)=>a.price.securePerVcpu-b.price.securePerVcpu)[0];
  if (!cpu) throw Error('No in-budget CPU backup capacity; checkpoint remains on network volume');
  const templates = unpack(await tools.mcp__runpod__list_public_templates({source:'official'}));
  const template = templates.templates.find(t=>t.id==='runpod-ubuntu-2404');
  if (!template) throw Error('Official CPU backup image missing');
  notify({phase:'backup_cpu_create',cpu:cpu.id,usd_per_hour:cpu.price.securePerVcpu*2,maximum_hours:4});
  const created = unpack(await tools.mcp__runpod__create_pod({body:{name:'qos-ai-557m-backup-20261019',cloud:'SECURE',
    cpu:{id:cpu.id,vcpuCount:2},dataCenterIds:['EU-RO-1'],image:template.image,disk:5,
    startSsh:true,startJupyter:false,ports:['22/tcp'],mounts:{network:[{volumeId:state.volume_id,path:'/workspace'}]}}}));
  backupId = created.id || created.pod?.id;
  if(!backupId) throw Error('CPU create returned no id');
  notify({phase:'backup_cpu_created',pod_id:backupId,volume_id:state.volume_id});
  }
  store('qos557_backup_pod_id',backupId);
  const backupDeadline = Date.now()/1000+4*3600;
  let n = null;
  for(let attempt=0;attempt<40;attempt++) {
    const p = unpack(await tools.mcp__runpod__get_pod({id:backupId}));
    n=network(p);if(n)break;
    await new Promise(r=>setTimeout(r,15000));
  }
  if(!n) throw Error('CPU backup pod never became SSH-ready');
  const scp = `scp -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -i ~/.ssh/id_ed25519 -i ~/.ssh/id_ed25519_hyperswine -P ${n.port}`;
  await finishCommand(await shell(`mkdir -p '${evidence}/remote' '${base}/qos_ai/runpod_557m_20261019'`, 'Prepare local training backup folders'),backupDeadline);
  const remote = `root@${n.ip}:/workspace/qos_ai_557m/experiment`;
  // Copy metadata and generated answers first; all binary artifacts follow.
  await finishCommand(await shell(scp+` -r ${remote}/evaluation '${evidence}/remote/'`, 'Back up the generated development benchmark responses'),backupDeadline);
  for(const name of ['student.pt','pretrained.pt','progress_pretrain.pt']) {
    await finishCommand(await shell(scp+` ${remote}/${name} '${base}/qos_ai/runpod_557m_20261019/${name}'`, 'Back up completed training weights and recovery state'),backupDeadline);
  }
  await finishCommand(await shell(scp+` '${remote}/*.json' '${remote}/*.jsonl' '${evidence}/remote/'`, 'Back up training manifests and learning curves'),backupDeadline);
  const metadata = `root@${n.ip}:/workspace/qos_ai_557m`;
  await finishCommand(await shell(`mkdir -p '${evidence}/remote/corpus' '${evidence}/remote/setup'`, 'Prepare reproducibility metadata backup folders'),backupDeadline);
  for(const name of ['dataset.json','documents.jsonl.gz']) {
    await finishCommand(await shell(scp+` ${metadata}/pretraining_data/${name} '${evidence}/remote/corpus/'`, 'Back up pinned corpus identities and accepted document ledger'),backupDeadline);
  }
  for(const name of ['budget.json','bundle_manifest.json','io_fix.json','environment.txt','tests.log','setup.log','job.log']) {
    await finishCommand(await shell(scp+` ${metadata}/${name} '${evidence}/remote/setup/'`, 'Back up the tested runtime and exact uploaded code manifest'),backupDeadline);
  }
  for(const name of ['training.log','prepare.log']) {
    await finishCommand(await shell(scp+` ${remote}/${name} '${evidence}/remote/setup/'`, 'Back up raw training and preparation logs'),backupDeadline);
  }
  const verify = `.venv/bin/python - <<'PY'
from pathlib import Path
import json
from qos_ai_data import file_hash
root=Path('qos_ai/runpod_557m_20261019')
meta=Path('qos_ai_benchmark/evidence/runpod_557m_20261019/remote')
pre=json.loads((meta/'pretraining_result.json').read_text())
sft=json.loads((meta/'sft_result.json').read_text())
expected={'student.pt':sft['checkpoint_sha256'],'pretrained.pt':pre['checkpoint_sha256'],'progress_pretrain.pt':pre['progress_sha256']}
for name,sha in expected.items():
 assert file_hash(root/name)==sha,name
(root/'checksums.json').write_text(json.dumps(expected,indent=2))
print(json.dumps({'backup_verified':True,'checksums':expected}))
PY`;
  const verified = await finishCommand(await shell(verify, 'Verify all copied model and optimizer hashes before cleanup',2000),backupDeadline);
  store('qos557_backup_verified',verified); notify(verified);
  await tools.mcp__runpod__pod_action({id:backupId,body:{action:'terminate'}});
  const remaining = unpack(await tools.mcp__runpod__list_pods({limit:1000}));
  if(remaining.pods.some(p=>p.id===backupId))throw Error('CPU deletion not confirmed');
  await tools.mcp__runpod__pod_action({id:state.pod_id,body:{action:'terminate'}});
  const rest = unpack(await tools.mcp__runpod__list_pods({limit:1000}));
  if(rest.pods.some(p=>p.id===state.pod_id))throw Error('GPU deletion not confirmed');
  // Volume deletion is allowed only after every required artifact is verified.
  await tools.mcp__runpod__delete_network_volume({id:state.volume_id});
  const volumes=unpack(await tools.mcp__runpod__list_network_volumes({limit:1000}));
  if(volumes.networkVolumes.some(v=>v.id===state.volume_id))throw Error('Volume deletion not confirmed');
  store('qos557_controller_complete',{terminal,backup_verified:true,resources_removed:true});
  notify({phase:'complete',backup_verified:true,resources_removed:true});
  const resourceUpdate = `.venv/bin/python - <<'PY'
from pathlib import Path
import json
p=Path('qos_ai_benchmark/evidence/runpod_557m_20261019/resources.json')
r=json.loads(p.read_text());r.update(resources_removed=True,backup_verified=True,status='training and artifact backup complete; cloud resources removed')
p.write_text(json.dumps(r,indent=2)+'\\n')
PY`;
  await finishCommand(await shell(resourceUpdate, 'Record verified cloud cleanup'),Date.now()/1000+60);
  try {
    const graded=await finishCommand(await shell('.venv/bin/python qos_ai_runpod_finalize.py > qos_ai_benchmark/evidence/runpod_557m_20261019/finalize.log 2>&1',
      'Grade backed-up generated code on the authorized M1 using its frozen compiler and benchmarks'),Date.now()/1000+1800);
    store('qos557_m1_grading_complete',true);notify({m1_grading_complete:true});
  } catch(gradingError) {store('qos557_grading_error',String(gradingError));notify({m1_grading_error:String(gradingError),local_backups_verified:true,resources_removed:true});}
} catch(error) {
  store('qos557_controller_error',String(error)); notify({controller_error:String(error)});
  try {await stopOwned(state.pod_id);store('qos557_gpu_stopped_after_error',true);}catch(e){notify({stop_error:String(e)});}
  const backupId=load('qos557_backup_pod_id');
  if(backupId)try{await stopOwned(backupId);}catch(e){notify({backup_stop_error:String(e)});}
  notify({preserved_volume_id:state.volume_id,action:'artifacts preserved; no volume deletion after failed verification'});
}
