import socket
import sys

def test_port(ip, port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((ip, port))
        if result == 0:
            print(f"✅ Port {port} is OPEN on {ip}")
            return True
        else:
            print(f"❌ Port {port} is CLOSED or unreachable on {ip} (Error code: {result})")
            return False
    except Exception as e:
        print(f"⚠️ Error testing port {port}: {e}")
        return False
    finally:
        sock.close()

if __name__ == "__main__":
    remote_ip = "10.137.227.217"
    print(f"Testing connectivity to {remote_ip}...")
    
    p5000 = test_port(remote_ip, 5000)
    p5001 = test_port(remote_ip, 5001)
    
    if p5000 or p5001:
        print("\n🎉 Connection successful! You can now run the launcher.")
    else:
        print("\n🔴 Connection failed. Ensure the other PC is running in 'server' mode and firewall allows ports 5000/5001.")
