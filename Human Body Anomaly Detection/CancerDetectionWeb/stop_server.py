import os
import signal
import psutil

def find_and_kill_flask():
    """Find and kill Flask processes running on port 5000"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Look for Python processes running run.py
            if proc.info['name'] == 'python.exe' and any('run.py' in cmd for cmd in proc.info['cmdline'] if cmd):
                print(f"Killing Flask process with PID {proc.info['pid']}")
                os.kill(proc.info['pid'], signal.SIGTERM)
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

if __name__ == "__main__":
    if find_and_kill_flask():
        print("Flask server stopped successfully!")
    else:
        print("No Flask server found running run.py") 