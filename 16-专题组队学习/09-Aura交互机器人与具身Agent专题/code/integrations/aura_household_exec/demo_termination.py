"""Keep a successful demo alive for release/stability evidence and its report.

Native goals, physics, observations, failure and timeout stay intact. This is
a demo lifecycle extension, not the benchmark's unmodified stop protocol.
"""
def defer_success(terminated,truncated,info,allow_terminal):
    success=isinstance(info,dict) and bool(info.get('done',{}).get('success',False))
    deferred=bool(terminated and not truncated and success and not allow_terminal)
    return (False if deferred else terminated),truncated,deferred
