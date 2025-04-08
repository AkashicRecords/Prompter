"""
Unit tests for the websocket module.
"""

import pytest
import asyncio
import json
import ssl
from unittest.mock import patch, MagicMock, AsyncMock
from cryptography.fernet import Fernet

# Import will be available after implementation
with pytest.raises(ImportError):
    from prompter.websocket import SecureWebsocketServer, SecureWebsocketClient

# Mock websockets library
class MockWebsocket:
    def __init__(self):
        self.sent_messages = []
        self.closed = False
        
    async def send(self, message):
        self.sent_messages.append(message)
        
    async def recv(self):
        return json.dumps({"type": "test", "data": "test_data"})
    
    async def close(self):
        self.closed = True

# Test classes
@pytest.mark.asyncio
class TestSecureWebsocketServer:
    """Test the SecureWebsocketServer class."""
    
    @pytest.fixture
    def mock_server(self, ssl_cert_key, encryption_key):
        """Create a mock server instance."""
        cert_path, key_path = ssl_cert_key
        
        with patch("websockets.serve", new_callable=AsyncMock) as mock_serve:
            with patch("ssl.SSLContext") as mock_ssl_context:
                with patch("prompter.websocket.SecureWebsocketServer", create=True):
                    from prompter.websocket import SecureWebsocketServer
                    
                    server = SecureWebsocketServer(
                        host="localhost",
                        port=8765,
                        ssl_cert=cert_path,
                        ssl_key=key_path,
                        encryption_key=encryption_key
                    )
                    
                    server._serve = mock_serve
                    server._websocket = MockWebsocket()
                    yield server
    
    async def test_initialization(self, mock_server, ssl_cert_key, encryption_key):
        """Test server initialization."""
        cert_path, key_path = ssl_cert_key
        
        assert mock_server.host == "localhost"
        assert mock_server.port == 8765
        assert mock_server.ssl_cert == cert_path
        assert mock_server.ssl_key == key_path
        assert mock_server.encryption_key == encryption_key
        assert isinstance(mock_server._cipher, Fernet)
        assert isinstance(mock_server._handlers, dict)
    
    async def test_register_handler(self, mock_server):
        """Test registering message handlers."""
        # Create a test handler
        async def test_handler(data):
            return {"status": "ok"}
        
        # Register the handler
        mock_server.register_handler("test_type", test_handler)
        
        # Check if handler was registered
        assert "test_type" in mock_server._handlers
        assert mock_server._handlers["test_type"] == test_handler
    
    async def test_handle_client(self, mock_server):
        """Test handling client connections."""
        # Create a mock handler
        test_handler = AsyncMock(return_value={"status": "ok"})
        mock_server.register_handler("test", test_handler)
        
        # Test handling a client
        await mock_server._handle_client(mock_server._websocket, None)
        
        # Verify the handler was called
        test_handler.assert_called_once_with("test_data")
        
        # Verify a response was sent
        assert len(mock_server._websocket.sent_messages) > 0
    
    async def test_encrypt_decrypt(self, mock_server):
        """Test encryption and decryption of messages."""
        # Original message
        original_message = {"type": "test", "data": "secret"}
        json_message = json.dumps(original_message)
        
        # Encrypt
        encrypted = mock_server._encrypt(json_message)
        
        # Decrypt
        decrypted = mock_server._decrypt(encrypted)
        
        # Verify
        assert json.loads(decrypted) == original_message
    
    async def test_start_server(self, mock_server):
        """Test starting the server."""
        # Mock asyncio.get_event_loop
        with patch("asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value = AsyncMock()
            
            # Start the server
            mock_server.start()
            
            # Verify server was started with correct parameters
            mock_server._serve.assert_called_once()
            args, kwargs = mock_server._serve.call_args
            assert kwargs["host"] == "localhost"
            assert kwargs["port"] == 8765
            assert "ssl" in kwargs


@pytest.mark.asyncio
class TestSecureWebsocketClient:
    """Test the SecureWebsocketClient class."""
    
    @pytest.fixture
    def mock_client(self, encryption_key, ssl_context):
        """Create a mock client instance."""
        with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
            with patch("prompter.websocket.SecureWebsocketClient", create=True):
                from prompter.websocket import SecureWebsocketClient
                
                # Set up the mock connect to return a mock websocket
                mock_ws = MockWebsocket()
                mock_connect.return_value.__aenter__.return_value = mock_ws
                
                client = SecureWebsocketClient(
                    uri="wss://localhost:8765",
                    ssl=ssl_context,
                    encryption_key=encryption_key
                )
                
                client._connect = mock_connect
                client._websocket = mock_ws
                yield client
    
    async def test_initialization(self, mock_client, encryption_key, ssl_context):
        """Test client initialization."""
        assert mock_client.uri == "wss://localhost:8765"
        assert mock_client.ssl == ssl_context
        assert mock_client.encryption_key == encryption_key
        assert isinstance(mock_client._cipher, Fernet)
    
    async def test_connect(self, mock_client):
        """Test connecting to the server."""
        # Connect to the server
        await mock_client.connect()
        
        # Verify connect was called with correct parameters
        mock_client._connect.assert_called_once()
        args, kwargs = mock_client._connect.call_args
        assert args[0] == "wss://localhost:8765"
        assert "ssl" in kwargs
    
    async def test_send_receive(self, mock_client):
        """Test sending and receiving data."""
        # Connect to the server
        await mock_client.connect()
        
        # Send a message
        await mock_client.send({"type": "test", "data": "hello"})
        
        # Verify message was sent
        assert len(mock_client._websocket.sent_messages) > 0
        
        # Receive a message
        response = await mock_client.receive()
        
        # Verify message was received and decrypted
        assert response == {"type": "test", "data": "test_data"}
    
    async def test_close(self, mock_client):
        """Test closing the connection."""
        # Connect to the server
        await mock_client.connect()
        
        # Close the connection
        await mock_client.close()
        
        # Verify connection was closed
        assert mock_client._websocket.closed


@pytest.mark.integration
class TestWebsocketIntegration:
    """Integration tests for the websocket module."""
    
    @pytest.mark.asyncio
    async def test_client_server_communication(self):
        """Test communication between client and server."""
        # This test will be implemented after the module is created
        pass
    
    @pytest.mark.asyncio
    async def test_multiple_clients(self):
        """Test multiple clients connecting to the server."""
        # This test will be implemented after the module is created
        pass 