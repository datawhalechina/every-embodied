"""Local CPU verification; never SSH into someone else's workstation."""
import subprocess,sys
if __name__=='__main__':
    raise SystemExit(subprocess.call([sys.executable,'-m','unittest','discover','-s','tests','-v']))
