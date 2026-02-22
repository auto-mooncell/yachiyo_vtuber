import socket
import struct
import time
import numpy as np
import sounddevice as sd

# 指向 EasyVtuber 的控制中枢
UDP_IP = "127.0.0.1"
UDP_PORT = 11573
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 麦克风灵敏度 (觉得嘴张得不够大就调高)
SENSITIVITY = 15.0 

def send_osf_packet(mouth_open):
    data = [0.0] * 70  
    data[0] = time.time()
    
    # 控制嘴巴开合 (Index 17)
    data[17] = mouth_open 
    # 强制让她睁眼，保持高冷
    data[25] = 1.0  
    data[26] = 1.0  
    
    packet = struct.pack(f'<{len(data)}f', *data)
    sock.sendto(packet, (UDP_IP, UDP_PORT))

def audio_callback(indata, frames, time_info, status):
    # 提取音量并转为张嘴幅度 (0.0 到 1.0)
    volume_norm = np.linalg.norm(indata) * SENSITIVITY
    mouth_val = min(1.0, volume_norm)
    send_osf_packet(mouth_val)

print("🎙️ AI 灵魂注入中... 请对着麦克风大喊！")
with sd.InputStream(callback=audio_callback):
    while True:
        time.sleep(0.1)