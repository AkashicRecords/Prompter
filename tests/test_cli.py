"""
Unit tests for the CLI module.
"""

import pytest
import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock, call

# Import test utilities
from test_utils import (
    temp_dir, 
    temp_file, 
    create_validated_mock
)

# Import will be available after implementation
with pytest.raises(ImportError):
    from prompter.cli import PrompterCLI, CommandProcessor, InteractiveMode


class TestCommandProcessor:
    """Test the CommandProcessor class."""
    
    @pytest.fixture
    def command_processor(self):
        """Create a command processor instance."""
        with patch("prompter.cli.CommandProcessor", create=True) as MockProcessor:
            from prompter.cli import CommandProcessor
            
            processor = CommandProcessor()
            
            # Mock command handlers
            processor._commands = {
                "help": Mock(return_value={"status": "success", "message": "Help information"}),
                "pattern": Mock(return_value={"status": "success", "message": "Pattern command"}),
                "train": Mock(return_value={"status": "success", "message": "Training started"}),
                "exit": Mock(return_value={"status": "exit", "message": "Exiting"})
            }
            
            yield processor
    
    def test_initialization(self, command_processor):
        """Test command processor initialization."""
        assert isinstance(command_processor._commands, dict), "Commands should be stored in a dictionary"
        assert len(command_processor._commands) > 0, "Commands dictionary should not be empty"
    
    def test_register_command(self, command_processor):
        """Test registering a command."""
        # Create a test command
        test_command = Mock(return_value={"status": "success", "message": "Test command"})
        
        # Register the command
        command_processor.register_command("test", test_command)
        
        # Check if command was registered
        assert "test" in command_processor._commands, "Command should be registered"
        assert command_processor._commands["test"] == test_command, "Command handler should match"
        
        # Test with invalid command name
        with pytest.raises(TypeError):
            command_processor.register_command(123, test_command)
        
        # Test with invalid command handler
        with pytest.raises(TypeError):
            command_processor.register_command("invalid", "not a function")
    
    def test_process_command(self, command_processor):
        """Test processing a command."""
        # Process a valid command
        result = command_processor.process_command("help")
        
        # Check result
        assert result["status"] == "success", "Status should be success"
        assert result["message"] == "Help information", "Message should match expected"
        assert command_processor._commands["help"].call_count == 1, "Command handler should be called once"
        
        # Process command with arguments
        result = command_processor.process_command("pattern create --name test")
        
        # Check result
        assert result["status"] == "success", "Status should be success"
        assert result["message"] == "Pattern command", "Message should match expected"
        command_processor._commands["pattern"].assert_called_with("create", "--name", "test")
        
        # Process unknown command
        result = command_processor.process_command("unknown")
        
        # Check result
        assert result["status"] == "error", "Status should be error for unknown command"
        assert "unknown command" in result["message"].lower(), "Message should indicate unknown command"
        
        # Test with invalid command type
        with pytest.raises(TypeError):
            command_processor.process_command(123)
    
    def test_help_command(self, command_processor):
        """Test help command."""
        # Process help command
        result = command_processor.process_command("help")
        
        # Check result
        assert result["status"] == "success", "Status should be success"
        assert result["message"] == "Help information", "Message should match expected"
        
        # Process help for specific command
        result = command_processor.process_command("help pattern")
        
        # Check result
        assert result["status"] == "success", "Status should be success"
        assert result["message"] == "Help information", "Message should match expected"
        command_processor._commands["help"].assert_called_with("pattern")
    
    def test_exit_command(self, command_processor):
        """Test exit command."""
        # Process exit command
        result = command_processor.process_command("exit")
        
        # Check result
        assert result["status"] == "exit", "Status should be exit"
        assert result["message"] == "Exiting", "Message should indicate exiting"
        assert command_processor._commands["exit"].call_count > 0, "Exit handler should be called"


class TestInteractiveMode:
    """Test the InteractiveMode class."""
    
    @pytest.fixture
    def interactive_mode(self):
        """Create an interactive mode instance."""
        with patch("prompter.cli.InteractiveMode", create=True) as MockInteractive:
            from prompter.cli import InteractiveMode
            from prompter.cli import CommandProcessor
            
            # Mock command processor
            processor = Mock(spec=CommandProcessor)
            processor.process_command.side_effect = lambda cmd: (
                {"status": "exit", "message": "Exiting"} if cmd == "exit" else
                {"status": "success", "message": f"Processed: {cmd}"}
            )
            
            interactive = InteractiveMode(processor)
            
            yield interactive
    
    def test_initialization(self, interactive_mode):
        """Test interactive mode initialization."""
        assert hasattr(interactive_mode, "_processor"), "Should have a processor attribute"
        assert hasattr(interactive_mode, "_running"), "Should have a running attribute"
        assert interactive_mode._running is False, "Should not be running initially"
    
    def test_start_stop(self, interactive_mode):
        """Test starting and stopping interactive mode."""
        # Mock input to return exit after a few commands
        with patch("builtins.input", side_effect=["help", "pattern list", "exit"]):
            # Start interactive mode
            interactive_mode.start()
            
            # Check that it's not running after exit
            assert interactive_mode._running is False, "Should not be running after exit"
    
    def test_handle_command(self, interactive_mode):
        """Test handling a command."""
        # Handle a command
        result = interactive_mode._handle_command("help")
        
        # Check result
        assert result["status"] == "success", "Status should be success"
        assert result["message"] == "Processed: help", "Message should match expected"
        
        # Handle exit command
        result = interactive_mode._handle_command("exit")
        
        # Check result
        assert result["status"] == "exit", "Status should be exit"
        assert result["message"] == "Exiting", "Message should indicate exiting"
    
    def test_input_handling(self, interactive_mode):
        """Test handling user input."""
        # Mock input and process methods
        with patch("builtins.input", side_effect=["  help  ", "pattern list", "exit"]):
            with patch.object(interactive_mode, "_handle_command") as mock_handle:
                mock_handle.side_effect = [
                    {"status": "success", "message": "Help info"},
                    {"status": "success", "message": "Pattern list"},
                    {"status": "exit", "message": "Exiting"}
                ]
                
                # Start interactive mode
                interactive_mode.start()
                
                # Check that commands were processed correctly
                assert mock_handle.call_count == 3, "Should process 3 commands"
                mock_handle.assert_has_calls([
                    call("help"),  # Note: whitespace should be stripped
                    call("pattern list"),
                    call("exit")
                ])
    
    def test_keyboard_interrupt(self, interactive_mode):
        """Test handling keyboard interrupt."""
        # Mock input to raise KeyboardInterrupt
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            # Start interactive mode
            interactive_mode.start()
            
            # Check that it's not running after interrupt
            assert interactive_mode._running is False, "Should not be running after interrupt"


class TestPrompterCLI:
    """Test the PrompterCLI class."""
    
    @pytest.fixture
    def cli(self):
        """Create a CLI instance."""
        with patch("prompter.cli.PrompterCLI", create=True) as MockCLI:
            from prompter.cli import PrompterCLI
            
            cli = PrompterCLI()
            
            # Mock command processor
            cli._processor = Mock()
            cli._processor.process_command.side_effect = lambda cmd: (
                {"status": "exit", "message": "Exiting"} if cmd == "exit" else
                {"status": "success", "message": f"Processed: {cmd}"}
            )
            
            # Mock interactive mode
            cli._interactive = Mock()
            
            yield cli
    
    def test_initialization(self, cli):
        """Test CLI initialization."""
        assert hasattr(cli, "_processor"), "Should have a processor attribute"
        assert hasattr(cli, "_interactive"), "Should have an interactive attribute"
        assert hasattr(cli, "_config_dir"), "Should have a config directory attribute"
    
    def test_parse_args(self, cli):
        """Test parsing command line arguments."""
        # Test with command
        args = cli.parse_args(["pattern", "list"])
        
        assert args.command == "pattern", "Command should be 'pattern'"
        assert args.args == ["list"], "Args should contain ['list']"
        
        # Test with interactive flag
        args = cli.parse_args(["-i"])
        
        assert args.interactive is True, "Interactive flag should be True"
        
        # Test with no arguments (should default to interactive)
        args = cli.parse_args([])
        
        assert args.interactive is True, "Interactive flag should default to True"
    
    def test_run_with_command(self, cli):
        """Test running with a command."""
        # Mock sys.argv
        with patch("sys.argv", ["prompter", "pattern", "list"]):
            # Mock parse_args to return a mock namespace
            mock_args = Mock()
            mock_args.interactive = False
            mock_args.command = "pattern"
            mock_args.args = ["list"]
            
            with patch.object(cli, "parse_args", return_value=mock_args):
                # Run CLI
                cli.run()
                
                # Check that command was processed
                cli._processor.process_command.assert_called_once_with("pattern list")
    
    def test_run_interactive(self, cli):
        """Test running in interactive mode."""
        # Mock sys.argv
        with patch("sys.argv", ["prompter", "-i"]):
            # Mock parse_args to return a mock namespace
            mock_args = Mock()
            mock_args.interactive = True
            
            with patch.object(cli, "parse_args", return_value=mock_args):
                # Run CLI
                cli.run()
                
                # Check that interactive mode was started
                cli._interactive.start.assert_called_once()
    
    def test_config_directory(self, cli, temp_dir):
        """Test config directory creation."""
        # Mock home directory
        with patch("pathlib.Path.home", return_value=temp_dir):
            # Create a new CLI instance to trigger config directory creation
            with patch("prompter.cli.PrompterCLI", create=True) as MockCLI:
                from prompter.cli import PrompterCLI
                
                # Mock necessary methods to avoid side effects
                with patch.object(PrompterCLI, "__init__", return_value=None):
                    cli = PrompterCLI()
                    cli._config_dir = temp_dir / ".prompter"
                    
                    # Ensure config directory exists
                    cli._ensure_config_dir()
                    
                    # Check that directory was created
                    assert cli._config_dir.exists(), "Config directory should be created"
                    assert cli._config_dir.is_dir(), "Config directory should be a directory"


@pytest.mark.integration
class TestCLIIntegration:
    """Integration tests for the CLI module."""
    
    def test_end_to_end(self):
        """Test end-to-end CLI workflow."""
        # This test will be implemented after the module is created
        pass 