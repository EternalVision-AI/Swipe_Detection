import asyncio
import websockets

clients = set()

async def handler(websocket, path):
    """Handles new WebSocket connections."""
    clients.add(websocket)
    print(f"New client connected: {path}")

    try:
        async for message in websocket:
            print(f"Received from client: {message}")
            # Broadcast the received message to all other clients
            await broadcast_message(message, websocket)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        clients.remove(websocket)

async def broadcast_message(message, sender_websocket):
    """Broadcasts a received message to all connected clients except the sender."""
    for client in clients:
        if client != sender_websocket:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                clients.remove(client)

async def main():
    server = await websockets.serve(handler, "localhost", 8765)
    print("WebSocket server started on ws://localhost:8765")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())