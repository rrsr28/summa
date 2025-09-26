import ssl
import socket

HOST = '127.0.0.1'
PORT = 9003

context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
context.load_cert_chain(certfile="tc.crt", keyfile="tc.key")
context.load_verify_locations(cafile="ca.crt")
context.verify_mode = ssl.CERT_REQUIRED   # force client certificate

clients = {}   # dict: {commonName: ssl_conn}

with socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0) as sock:
    sock.bind((HOST, PORT))
    sock.listen(5)
    print(f"[TC] Listening on {HOST}:{PORT}")

    while True:
        conn, addr = sock.accept()
        ssl_conn = context.wrap_socket(conn, server_side=True)
        peer_cert = ssl_conn.getpeercert()
        cn = dict(x[0] for x in peer_cert['subject'])['commonName']
        clients[cn] = ssl_conn
        print(f"[TC] {cn} connected")

        def forward_messages(cn, ssl_conn):
            try:
                while True:
                    data = ssl_conn.recv(4096)
                    if not data:
                        break
                    print(f"[TC] from {cn}: {data.decode()}")
                    for other, other_conn in clients.items():
                        if other != cn:
                            other_conn.sendall(f"From {cn}: {data.decode()}".encode())
            except Exception as e:
                print(f"[TC] error with {cn}: {e}")
            finally:
                ssl_conn.close()
                del clients[cn]
                print(f"[TC] {cn} disconnected")

        # run each client in its own thread
        import threading
        threading.Thread(target=forward_messages, args=(cn, ssl_conn), daemon=True).start()
