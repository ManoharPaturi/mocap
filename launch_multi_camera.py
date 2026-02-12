"""
Multi-Camera Launcher
Easily switch between single, server, and master modes
"""

import sys
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Launch VS5 Multi-Camera System')
parser.add_argument('--mode', type=str, default='single',
                    choices=['single', 'server', 'master'],
                    help='Camera mode: single/server/master')
parser.add_argument('--remote-ip', type=str, default=None,
                    help='IP address of remote camera (for master mode)')
args = parser.parse_args()

# Update config dynamically
import config
config.MULTI_CAMERA_MODE = args.mode

if args.mode == 'master':
    if args.remote_ip:
        config.REMOTE_CAMERA_IP = args.remote_ip
    else:
        print("WARNING: Master mode requires --remote-ip argument!")
        print("Example: python launch_multi_camera.py --mode master --remote-ip 10.51.179.228")
        sys.exit(1)

# Launch main GUI
from main_gui import MocapGUI

print(f"\n{'='*60}")
print(f"VS5 Motion Capture System")
print(f"Mode: {args.mode.upper()}")
if args.mode == 'master':
    print(f"Remote Camera: {config.REMOTE_CAMERA_IP}")
print(f"{'='*60}\n")

app = MocapGUI()
app.run()
