import inference
import socket
import wave
import numpy as np

class AudioSocket:
    def __init__(self, ip, port, max_client):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.port = port
        self.server.bind((ip, port))
        self.server.listen(max_client)
        self.sample_rate = 32000
        self.block_length = self.sample_rate * 4

        self.id = 0

    def save(self, buf):
        wf = wave.open(str(self.id) +'.wav', 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(32000)
        wf.writeframes(buf)
        wf.close()
        self.id += 1

    def stream(self):
        print('Listening')
        buf = b''
        while True:
            (clientsocket, address) = self.server.accept()
            print('Connected', address)
            while True:
                chunk = clientsocket.recv(1024)
                if chunk == b'':
                    print('Disconnected', address)
                    break
                else:
                    buf += chunk
                if len(buf) >= self.block_length:
                    yield buf[0:self.block_length]
                    buf = buf[self.block_length:]

if __name__ == '__main__':
#infer = inference.Inference('./checkpoint/checkpoint_frame_batch_64/iter_5000.pth')
    infer = inference.Inference('./checkpoint/checkpoint_batch_128/iter_13000.pth')
    server = AudioSocket('0.0.0.0', 8001, 1)
    for block in server.stream():
        print(len(block))
#server.save(block)
        block = np.fromstring(block, dtype=np.short) / 32768.
        infer.inference(None, block)
