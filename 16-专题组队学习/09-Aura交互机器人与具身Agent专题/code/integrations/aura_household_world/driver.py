"""Future dedicated bounded driver; importing this file never opens a socket."""
import argparse,asyncio,os
from pathlib import Path
import threading


def main():
    p=argparse.ArgumentParser();p.add_argument('--policy-port',type=int,required=True)
    p.add_argument('--socket',type=Path,required=True);p.add_argument('--status',type=Path,required=True)
    p.add_argument('--cap',type=int,required=True);a=p.parse_args()
    if a.cap<100:raise ValueError('Initialization and terminal observation require a bounded budget')
    from tools.util.driver import Driver,ControlServer,ControlHandler,serve
    class BoundedDriver(Driver):
        def request(self,payload):
            with self.condition:
                if payload.get('operation')=='step' and self.ticks>=a.cap-2:
                    raise ValueError('Reserved native budget for final observation and terminal')
            return super().request(payload)
    if a.socket.exists():raise FileExistsError(a.socket)
    a.socket.parent.mkdir(parents=True,exist_ok=False);a.status.parent.mkdir(parents=True,exist_ok=True)
    driver=BoundedDriver(a.status)
    with ControlServer(str(a.socket),ControlHandler) as server:
        os.chmod(a.socket,0o600);server.driver=driver
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:asyncio.run(serve(driver,'127.0.0.1',a.policy_port))
        finally:driver.disconnected();server.shutdown()

if __name__=='__main__':main()
