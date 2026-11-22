"""Adapt storage accounting to a fresh private appdata path on shared hosts.

An opaque older process cannot have inherited a newly generated UUID path.
Each such process receives an additional cache-sized disk margin. The original
reservation, visible-path collision checks and disk watchdog remain active;
automatic deletion of other runs' caches is disabled for this private path.
"""
import fcntl
import json
import os
from pathlib import Path
import stat
import time
import uuid

import psutil

PREEXISTING={}


def private_appdata(base):
    """Create a mode-0700 random leaf and hold its exclusive lifetime lease."""
    base=Path(base).resolve()
    base.mkdir(parents=True,exist_ok=True)
    if base.stat().st_uid!=os.getuid():raise ValueError('Private appdata parent belongs to another user')
    before=time.time()
    identities={}
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit():continue
        try:
            if folder.stat().st_uid==os.getuid():
                process=psutil.Process(int(folder.name));identities[process.pid]=process.create_time()
        except (FileNotFoundError,ProcessLookupError,psutil.NoSuchProcess):pass
    path=base/('owner-'+str(os.getpid())+'-'+uuid.uuid4().hex)
    path.mkdir(mode=0o700)
    lease=(path/'.owner.lock').open('x')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if stat.S_IMODE(path.stat().st_mode)!=0o700:raise ValueError('Private appdata permissions changed')
    PREEXISTING[str(path)]=identities
    return path,lease,before


def opaque_older_processes(before):
    """Collect only identity metadata and open permissions; read no environment."""
    result={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            if path.stat().st_uid!=os.getuid():continue
            process=psutil.Process(int(path.name))
            if process.status()==psutil.STATUS_ZOMBIE or process.create_time()>before-2:continue
            try:
                descriptor=os.open(path/'environ',os.O_RDONLY|os.O_CLOEXEC)
                os.close(descriptor)
            except PermissionError:
                result[process.pid]={'birth':process.create_time(),'name':process.name(),'uid':os.getuid()}
        except (FileNotFoundError,ProcessLookupError,psutil.NoSuchProcess):pass
    return result


def still_same_older(pid,rows,before,identity=None):
    if pid not in rows:return False
    try:
        process=(identity or psutil.Process)(pid)
        birth=process.create_time()
        return (abs(birth-rows[pid]['birth'])<.02 and (before is None or birth<=before-2)
                and process.status()!=psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:return False


def storage_budget(storage,opaque_count):
    cache=storage._env_float(storage.CACHE_MAX_ENV,16.0)
    hard=storage._env_float(storage.HARD_FREE_ENV,64.0)
    if cache<=0 or hard<=0:raise ValueError('Private appdata requires bounded cache and positive disk reserve')
    return {'cache_max_gib':cache,'hard_free_gib':hard+opaque_count*cache,'soft_free_gib':0.0}


def run_private_preflight(storage,original,path,lease,before,*args,**kwargs):
    """Retain stock storage guards with conservative opaque-process capacity."""
    if Path(args[0] if args else kwargs.get('appdata_path','')).resolve()!=path:
        raise ValueError('Storage check must target this process fresh private directory')
    if kwargs.get('reserve_for_pid')!=os.getpid():raise ValueError('Only the current evaluator may reserve its private path')
    if lease.closed or stat.S_IMODE(path.stat().st_mode)!=0o700:
        raise ValueError('Private appdata lease is unavailable')
    rows=opaque_older_processes(before)
    identities=PREEXISTING.get(str(path))
    if identities is not None:
        rows={}
        for pid,birth in identities.items():
            try:
                process=psutil.Process(pid)
                if abs(process.create_time()-birth)>.02 or process.status()==psutil.STATUS_ZOMBIE:continue
                try:
                    descriptor=os.open(f'/proc/{pid}/environ',os.O_RDONLY|os.O_CLOEXEC);os.close(descriptor)
                except PermissionError:rows[pid]={'birth':birth,'name':process.name(),'uid':os.getuid()}
            except (FileNotFoundError,ProcessLookupError,psutil.NoSuchProcess):pass
    budget=storage_budget(storage,len(rows))
    classifier=storage._process_may_use_appdata
    excluded=set()
    def classify(pid):
        if still_same_older(pid,rows,None if identities is not None else before):
            excluded.add(pid)
            return False
        return classifier(pid)
    storage._process_may_use_appdata=classify
    try:
        report=original(*args,**{**kwargs,**budget})
    finally:
        storage._process_may_use_appdata=classifier
    report['private_appdata_accounting']={'fresh_path':str(path),'exclusive_lease':True,
        'opaque_older_processes_budgeted':len(rows),'opaque_processes_excluded_from_collision_scan':len(excluded),
        'extra_reserve_gib':len(rows)*budget['cache_max_gib'],'hard_free_gib':budget['hard_free_gib'],
        'identity_snapshot_precedes_private_directory':identities is not None,
        'other_cache_cleanup':False,'argument_bytes_read_by_adapter':0,'environment_bytes_read_by_adapter':0}
    with (path/'private-storage-accounting.json').open('x') as stream:json.dump(report['private_appdata_accounting'],stream,indent=2)
    print('[private-storage] '+json.dumps(report['private_appdata_accounting']),flush=True)
    return report
