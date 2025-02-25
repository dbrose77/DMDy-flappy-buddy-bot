import asyncio
import json
import sys
import signal
import websockets

from Bots.aiFactory import ai_factory
from Bots.data import PlayState

# Global bot reference for shutdown handling
current_bot = None

def handle_shutdown(signum, frame):
    """Handle shutdown gracefully"""
    print("\nReceived shutdown signal...")
    if current_bot and hasattr(current_bot, 'shutdown'):
        current_bot.shutdown()
    sys.exit(0)

async def handle_message(bot, websocket):
    response_as_string = await websocket.recv()
    try:
        # Check if response is empty
        if not response_as_string:
            print("Received empty response from server")
            return
            
        # Try to decode the response
        try:
            decoded_response = response_as_string.decode('utf-8')
        except UnicodeDecodeError:
            print("Received ping message")
            return
            
        # Skip empty or invalid JSON
        if not decoded_response or decoded_response.isspace():
            print("Received empty decoded response")
            return
            
        # Replace single quotes with double quotes for JSON parsing
        decoded_response = decoded_response.replace("'", '"')
        
        try:
            # Parse JSON and create PlayState
            json_data = json.loads(decoded_response)
            response: PlayState = PlayState.from_dict(json_data)
            
            # Get bot's action
            bot_answer = bot.play(response)
            
            # Send response back to server
            await websocket.send(json.dumps({"fly": bot_answer}))
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Received data: {decoded_response}")
        except Exception as e:
            print(f"Error processing message: {e}")
            
    except Exception as e:
        print(f"Unexpected error in handle_message: {e}")


async def client(bot, port):
    global current_bot
    current_bot = bot
    
    uri = f"ws://localhost:{port}/{bot.get_name()}"
    try:
        async with websockets.connect(uri, ping_timeout=None, ping_interval=None) as websocket:
            print("Connected to server.")
            while True:
                try:
                    await handle_message(bot, websocket)
                except websockets.ConnectionClosedOK:
                    print("Connection closed by server.")
                    break
                except websockets.ConnectionClosedError:
                    print("Server was shut down.")
                    break
    finally:
        if hasattr(bot, 'shutdown'):
            bot.shutdown()


if __name__ == "__main__":
    # Register shutdown handlers
    signal.signal(signal.SIGINT, handle_shutdown)  # Handles Ctrl+C
    signal.signal(signal.SIGTERM, handle_shutdown)  # Handles termination signal
    
    ai_bot = ai_factory()
    used_port = 5050
    if len(sys.argv) == 2:
        ai_bot = ai_factory(sys.argv[1])
    if len(sys.argv) == 3:
        ai_bot = ai_factory(sys.argv[1])
        used_port = sys.argv[2]

    try:
        asyncio.run(client(ai_bot, used_port))
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt...")
    finally:
        if hasattr(ai_bot, 'shutdown'):
            ai_bot.shutdown()
