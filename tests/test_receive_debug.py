"""
Debug test - Check if data is being received
"""

import zmq
import msgpack
import time

print("Testing direct ZMQ connection to PC1...")
print("Connecting to 10.12.74.224:5001")

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://10.12.74.224:5001")
socket.setsockopt(zmq.SUBSCRIBE, b"")

print("\nListening for data for 10 seconds...")

received = 0
start_time = time.time()

while time.time() - start_time < 10:
    try:
        if socket.poll(timeout=1000):  # 1 second timeout
            data = socket.recv()
            received += 1
            
            # Try to decode
            try:
                msg = msgpack.unpackb(data)
                print(f"✅ Received message #{received}: {msg.get('type', 'unknown')}")
            except:
                print(f"✅ Received raw data #{received} ({len(data)} bytes)")
    except Exception as e:
        print(f"❌ Error: {e}")

print(f"\n{'✅' if received > 0 else '❌'} Total messages received: {received}")

socket.close()
context.term()
