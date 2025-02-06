# server.py
import socket

try:
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Socket created successfully.")
except Exception as e:
    print(f"Failed to create socket: {e}")
    exit()

try:
    # Get the local machine name and define a port
    host = '0.0.0.0'  # Listen on all available interfaces
    port = 12345       # Port to listen on

    # Bind the socket to the port
    server_socket.bind((host, port))
    print(f"Socket bound to {host}:{port}.")
except Exception as e:
    print(f"Failed to bind socket: {e}")
    server_socket.close()
    exit()

try:
    # Listen for incoming connections (max 1 connection)
    server_socket.listen(1)
    print(f"Server listening on {host}:{port}...")
except Exception as e:
    print(f"Failed to listen: {e}")
    server_socket.close()
    exit()

try:
    # Accept a connection
    print("Waiting for a connection...")
    client_socket, addr = server_socket.accept()
    print(f"Connection established with {addr}")

    # Receive data from the client
    print("Waiting for data...")
    data = client_socket.recv(1024).decode('utf-8')
    print(f"Received message: {data}")

    # Close the connection
    client_socket.close()
    server_socket.close()
except Exception as e:
    print(f"Error during connection: {e}")
    server_socket.close()