"""Private observer proof for the official three-can goal, never model input."""
TARGETS={'can_of_soda_113','can_of_soda_114','can_of_soda_115'}

def verify(observer,status):
    tick=observer.get('wrapper_tick')
    if (status.get('connected') is not True or type(tick) is not int
        or not 0<=status.get('ticks',-1)-tick<=15):return None
    inside=observer.get('inside_bin_observer_only',{})
    stable=observer.get('stable_released_samples',{})
    held=set(observer.get('assisted_grasp_targets_observer_only',{}).values())
    if not all(inside.get(n) is True and type(stable.get(n)) is int and stable[n]>=2 and n not in held for n in TARGETS):return None
    progress=observer.get('progress',{})
    if progress.get('ok') is not True or progress.get('complete') is not True:return None
    return {'passed':True,'scope':'three_cans','episode_id':status['episode_id'],'native_tick':tick,
            'released':True,'inside':sorted(TARGETS),'stable_native_observer_samples':min(stable[n] for n in TARGETS),
            'official_task_complete':True,'official_progress':progress,
            'source':'native Inside, released assisted grasp, low object speed across consecutive samples; official predicate complete'}
