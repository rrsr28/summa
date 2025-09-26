import ssl
import socket

HOST = '127.0.0.1'
PORT = 9003

context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile="ca.crt")
context.load_cert_chain(certfile="b.crt", keyfile="b.key")

with socket.create_connection((HOST, PORT)) as sock:
    with context.wrap_socket(sock, server_hostname="TC") as ssock:
        print("[B] Connected to TC")
        while True:
            data = ssock.recv(4096)
            if not data:
                break
            print(f"[B] Received: {data.decode()}")
            reply = input("Reply to A: ")
            ssock.sendall(reply.encode())
