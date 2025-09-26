import ssl
import socket

HOST = '127.0.0.1'
PORT = 9003

context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile="ca.crt")
context.load_cert_chain(certfile="a.crt", keyfile="a.key")

with socket.create_connection((HOST, PORT)) as sock:
    with context.wrap_socket(sock, server_hostname="TC") as ssock:
        print("[A] Connected to TC")
        while True:
            msg = input("Message to B: ")
            ssock.sendall(msg.encode())
            data = ssock.recv(4096)
            print(f"[A] Received: {data.decode()}")
