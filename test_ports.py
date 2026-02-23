"""
Simple network connectivity test
Tests if PC2 can connect to PC1's ports
"""

import socket

pc1_ip = "10.137.227.228"

print(f"Testing connectivity to {pc1_ip}...")

# Test discovery port (5000)
print("\n1. Testing port 5000 (discovery)...")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)
    result = sock.connect_ex((pc1_ip, 5000))
    if result == 0:
        print(f"   ✅ Port 5000 is OPEN")
    else:
        print(f"   ❌ Port 5000 is CLOSED or BLOCKED")
    sock.close()
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test data port (5001)
print("\n2. Testing port 5001 (data)...")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)
    result = sock.connect_ex((pc1_ip, 5001))
    if result == 0:
        print(f"   ✅ Port 5001 is OPEN")
    else:
        print(f"   ❌ Port 5001 is CLOSED or BLOCKED")
    sock.close()
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)
print("If both ports show OPEN: Network is fine, check code")
print("If ports show CLOSED: Firewall is blocking")
print("If you get connection errors: Wrong IP or network issue")
