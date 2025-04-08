"""
Websocket module for Prompter.

This module provides secure websocket communication for the Prompter package.
"""

import asyncio
import json
import ssl
import websockets
from typing import Dict, Any, Callable, Optional, List
from pathlib import Path
from cryptography.fernet import Fernet
from .logging import logger

class SecureWebsocketServer:
    """Secure websocket server for Prompter."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        ssl_cert: Optional[Path] = None,
        ssl_key: Optional[Path] = None,
        encryption_key: Optional[bytes] = None
    ):
        """Initialize the websocket server.
        
        Args:
            host: The host to bind to.
            port: The port to bind to.
            ssl_cert: Path to the SSL certificate.
            ssl_key: Path to the SSL key.
            encryption_key: Key for encryption. If None, a new key will be generated.
        """
        self.host = host
        self.port = port
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        self.handlers: Dict[str, Callable] = {}
        self.clients: List[websockets.WebSocketServerProtocol] = []
        
        # Set up SSL context if certificates are provided
        self.ssl_context = None
        if ssl_cert and ssl_key:
            self.ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            self.ssl_context.load_cert_chain(ssl_cert, ssl_key)
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register a handler for a message type.
        
        Args:
            message_type: The type of message to handle.
            handler: The handler function.
        """
        self.handlers[message_type] = handler
    
    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle a client connection.
        
        Args:
            websocket: The websocket connection.
            path: The path of the request.
        """
        self.clients.append(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    # Decrypt the message
                    decrypted_message = self.fernet.decrypt(message.encode())
                    data = json.loads(decrypted_message)
                    
                    # Handle the message
                    message_type = data.get("type")
                    if message_type in self.handlers:
                        response = await self.handlers[message_type](data, websocket)
                        
                        # Encrypt and send the response
                        if response:
                            encrypted_response = self.fernet.encrypt(json.dumps(response).encode())
                            await websocket.send(encrypted_response)
                    else:
                        logger.warning(f"Unknown message type: {message_type}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        finally:
            self.clients.remove(websocket)
    
    async def start(self):
        """Start the websocket server."""
        server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ssl=self.ssl_context
        )
        
        logger.info(f"Websocket server started on {self.host}:{self.port}")
        await server.wait_closed()
    
    def run(self):
        """Run the websocket server."""
        asyncio.run(self.start())

class SecureWebsocketClient:
    """Secure websocket client for Prompter."""
    
    def __init__(
        self,
        uri: str = "ws://localhost:8765",
        ssl_verify: bool = True,
        encryption_key: Optional[bytes] = None
    ):
        """Initialize the websocket client.
        
        Args:
            uri: The URI of the websocket server.
            ssl_verify: Whether to verify SSL certificates.
            encryption_key: Key for encryption.
        """
        self.uri = uri
        self.ssl_verify = ssl_verify
        self.encryption_key = encryption_key
        self.fernet = Fernet(encryption_key) if encryption_key else None
        self.websocket = None
        
        # Set up SSL context
        self.ssl_context = None
        if uri.startswith("wss://"):
            self.ssl_context = ssl.create_default_context()
            if not ssl_verify:
                self.ssl_context.check_hostname = False
                self.ssl_context.verify_mode = ssl.CERT_NONE
    
    async def connect(self):
        """Connect to the websocket server."""
        self.websocket = await websockets.connect(
            self.uri,
            ssl=self.ssl_context
        )
        logger.info(f"Connected to {self.uri}")
    
    async def send(self, data: Dict[str, Any]):
        """Send data to the server.
        
        Args:
            data: The data to send.
        """
        if not self.websocket:
            raise RuntimeError("Not connected to server")
        
        # Encrypt the data
        if self.fernet:
            encrypted_data = self.fernet.encrypt(json.dumps(data).encode())
            await self.websocket.send(encrypted_data)
        else:
            await self.websocket.send(json.dumps(data))
    
    async def receive(self) -> Dict[str, Any]:
        """Receive data from the server.
        
        Returns:
            The received data.
        """
        if not self.websocket:
            raise RuntimeError("Not connected to server")
        
        message = await self.websocket.recv()
        
        # Decrypt the message
        if self.fernet:
            decrypted_message = self.fernet.decrypt(message.encode())
            return json.loads(decrypted_message)
        else:
            return json.loads(message)
    
    async def close(self):
        """Close the connection."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            logger.info("Disconnected from server") 