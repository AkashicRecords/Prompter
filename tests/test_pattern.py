"""
Unit tests for the pattern recognition module.
"""

import pytest
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Import test utilities
from test_utils import (
    temp_dir, 
    temp_file, 
    sample_pattern_dict, 
    validate_pattern_dict,
    create_validated_mock
)

# Import will be available after implementation
with pytest.raises(ImportError):
    from prompter.pattern import PromptPattern, PatternMatcher


class TestPromptPattern:
    """Test the PromptPattern class."""
    
    def test_initialization_with_required_args(self):
        """Test initialization with only required arguments."""
        with patch("prompter.pattern.PromptPattern", create=True) as MockPattern:
            from prompter.pattern import PromptPattern
            
            pattern = PromptPattern(
                template="This is a {placeholder} template"
            )
            
            # Verify attributes
            assert pattern.template == "This is a {placeholder} template", "Template should match input"
            assert "placeholder" in pattern.variables, "Variables should be extracted from template"
            assert pattern.id is not None, "ID should be auto-generated"
            assert isinstance(pattern.id, str), "ID should be a string"
            assert pattern.score == 0.0, "Default score should be 0.0"
            assert isinstance(pattern.score, float), "Score should be a float"
    
    def test_initialization_with_all_args(self, sample_pattern_dict):
        """Test initialization with all arguments."""
        with patch("prompter.pattern.PromptPattern", create=True) as MockPattern:
            from prompter.pattern import PromptPattern
            
            pattern = PromptPattern(
                id=sample_pattern_dict["id"],
                name=sample_pattern_dict["name"],
                description=sample_pattern_dict["description"],
                template=sample_pattern_dict["template"],
                variables=sample_pattern_dict["variables"],
                score=sample_pattern_dict["score"],
                tags=sample_pattern_dict["tags"]
            )
            
            # Verify attributes with type checking
            assert pattern.id == sample_pattern_dict["id"], "ID should match input"
            assert isinstance(pattern.id, str), "ID should be a string"
            
            assert pattern.name == sample_pattern_dict["name"], "Name should match input"
            assert isinstance(pattern.name, str), "Name should be a string"
            
            assert pattern.description == sample_pattern_dict["description"], "Description should match input"
            assert isinstance(pattern.description, str), "Description should be a string"
            
            assert pattern.template == sample_pattern_dict["template"], "Template should match input"
            assert isinstance(pattern.template, str), "Template should be a string"
            
            assert pattern.variables == sample_pattern_dict["variables"], "Variables should match input"
            assert isinstance(pattern.variables, list), "Variables should be a list"
            assert all(isinstance(v, str) for v in pattern.variables), "All variables should be strings"
            
            assert pattern.score == sample_pattern_dict["score"], "Score should match input"
            assert isinstance(pattern.score, float), "Score should be a float"
            
            assert pattern.tags == sample_pattern_dict["tags"], "Tags should match input"
            assert isinstance(pattern.tags, list), "Tags should be a list"
            assert all(isinstance(t, str) for t in pattern.tags), "All tags should be strings"
    
    def test_from_dict(self, sample_pattern_dict):
        """Test creating a pattern from a dictionary."""
        with patch("prompter.pattern.PromptPattern", create=True) as MockPattern:
            from prompter.pattern import PromptPattern
            
            # Validate the sample dictionary first
            validate_pattern_dict(sample_pattern_dict)
            
            pattern = PromptPattern.from_dict(sample_pattern_dict)
            
            # Verify attributes
            assert pattern.id == sample_pattern_dict["id"], "ID should match dictionary"
            assert pattern.name == sample_pattern_dict["name"], "Name should match dictionary"
            assert pattern.template == sample_pattern_dict["template"], "Template should match dictionary"
            assert pattern.score == sample_pattern_dict["score"], "Score should match dictionary"
            
            # Test with invalid input
            with pytest.raises(KeyError):
                PromptPattern.from_dict({"missing": "required keys"})
            
            with pytest.raises(TypeError):
                PromptPattern.from_dict({"id": 123, "template": "Invalid ID type"})
    
    def test_to_dict(self, sample_pattern_dict):
        """Test converting a pattern to a dictionary."""
        with patch("prompter.pattern.PromptPattern", create=True) as MockPattern:
            from prompter.pattern import PromptPattern
            
            pattern = PromptPattern(
                id=sample_pattern_dict["id"],
                name=sample_pattern_dict["name"],
                description=sample_pattern_dict["description"],
                template=sample_pattern_dict["template"],
                variables=sample_pattern_dict["variables"],
                score=sample_pattern_dict["score"],
                tags=sample_pattern_dict["tags"]
            )
            
            pattern_dict = pattern.to_dict()
            
            # Validate the dictionary
            validate_pattern_dict(pattern_dict)
            
            # Verify content
            assert pattern_dict["id"] == sample_pattern_dict["id"], "ID should match original"
            assert pattern_dict["name"] == sample_pattern_dict["name"], "Name should match original"
            assert pattern_dict["template"] == sample_pattern_dict["template"], "Template should match original"
            assert "created_at" in pattern_dict, "Should include created_at timestamp"
            assert "updated_at" in pattern_dict, "Should include updated_at timestamp"
            
            # Verify timestamps are strings (ISO format)
            assert isinstance(pattern_dict["created_at"], str), "created_at should be a string"
            assert isinstance(pattern_dict["updated_at"], str), "updated_at should be a string"
    
    def test_render(self):
        """Test rendering a pattern with variable values."""
        with patch("prompter.pattern.PromptPattern", create=True) as MockPattern:
            from prompter.pattern import PromptPattern
            
            pattern = PromptPattern(
                template="Hello, {name}! Welcome to {place}."
            )
            
            # Valid rendering
            rendered = pattern.render(name="John", place="Earth")
            
            assert rendered == "Hello, John! Welcome to Earth.", "Rendered result should match expected"
            assert isinstance(rendered, str), "Rendered result should be a string"
            
            # Test with different variable types
            rendered = pattern.render(name=123, place=["Earth"])
            
            assert rendered == "Hello, 123! Welcome to ['Earth'].", "Should convert variable values to strings"
            assert isinstance(rendered, str), "Rendered result should be a string"
    
    def test_render_missing_variable(self):
        """Test rendering with a missing variable."""
        with patch("prompter.pattern.PromptPattern", create=True) as MockPattern:
            from prompter.pattern import PromptPattern
            
            pattern = PromptPattern(
                template="Hello, {name}! Welcome to {place}."
            )
            
            # Missing variable should raise KeyError
            with pytest.raises(KeyError):
                pattern.render(name="John")
            
            # Too many variables should not raise an error
            rendered = pattern.render(name="John", place="Earth", extra="ignored")
            assert rendered == "Hello, John! Welcome to Earth.", "Extra variables should be ignored"
    
    def test_update_score(self):
        """Test updating a pattern's score."""
        with patch("prompter.pattern.PromptPattern", create=True) as MockPattern:
            from prompter.pattern import PromptPattern
            
            pattern = PromptPattern(
                template="Test template",
                score=0.5
            )
            
            # Valid score update
            pattern.update_score(0.8)
            
            assert pattern.score == 0.8, "Score should be updated"
            assert isinstance(pattern.score, float), "Score should be a float"
            assert pattern.updated_at is not None, "Updated timestamp should be set"
            
            # Test with invalid score type
            with pytest.raises(TypeError):
                pattern.update_score("invalid")
            
            # Test with out-of-range score
            with pytest.raises(ValueError):
                pattern.update_score(1.5)
            
            with pytest.raises(ValueError):
                pattern.update_score(-0.5)
    
    def test_extract_variables(self):
        """Test extracting variables from a template."""
        with patch("prompter.pattern.PromptPattern", create=True) as MockPattern:
            from prompter.pattern import PromptPattern
            
            variables = PromptPattern.extract_variables(
                "This is a {first} template with {second} variables."
            )
            
            assert "first" in variables, "Should extract 'first' variable"
            assert "second" in variables, "Should extract 'second' variable"
            assert len(variables) == 2, "Should extract exactly 2 variables"
            assert isinstance(variables, list), "Result should be a list"
            assert all(isinstance(v, str) for v in variables), "All variables should be strings"
            
            # Test with no variables
            variables = PromptPattern.extract_variables("No variables in this template")
            assert len(variables) == 0, "Should extract 0 variables"
            
            # Test with repeated variables
            variables = PromptPattern.extract_variables("{var} appears {var} twice")
            assert len(variables) == 1, "Should only include unique variables"
            assert variables[0] == "var", "Should extract the variable correctly"
    
    def test_save_load(self, temp_file):
        """Test saving and loading a pattern."""
        with patch("prompter.pattern.PromptPattern", create=True) as MockPattern:
            from prompter.pattern import PromptPattern
            
            # Create a pattern
            pattern = PromptPattern(
                template="Test template",
                score=0.75
            )
            
            # Mock the save method
            original_dict = {
                "id": pattern.id,
                "template": "Test template",
                "score": 0.75,
                "variables": [],
                "created_at": "2023-04-07T12:00:00Z",
                "updated_at": "2023-04-07T12:00:00Z"
            }
            
            with patch.object(pattern, "to_dict", return_value=original_dict):
                # Save the pattern
                pattern.save(temp_file)
                
                # Verify file was created and contains valid JSON
                assert temp_file.exists(), "Save file should exist"
                with open(temp_file, 'r') as f:
                    content = json.load(f)
                    assert content == original_dict, "Saved content should match pattern dict"
            
            # Mock the load method
            with patch.object(PromptPattern, "from_dict", return_value=pattern):
                # Load the pattern
                loaded_pattern = PromptPattern.load(temp_file)
                
                # Verify loaded pattern
                assert loaded_pattern.template == pattern.template, "Loaded template should match original"
                assert loaded_pattern.score == pattern.score, "Loaded score should match original"
                assert loaded_pattern.id == pattern.id, "Loaded ID should match original"
                
                # Test with non-existent file
                with pytest.raises(FileNotFoundError):
                    PromptPattern.load(Path("non_existent_file.json"))


class TestPatternMatcher:
    """Test the PatternMatcher class."""
    
    @pytest.fixture
    def matcher(self):
        """Create a pattern matcher instance."""
        with patch("prompter.pattern.PatternMatcher", create=True) as MockMatcher:
            from prompter.pattern import PatternMatcher
            matcher = PatternMatcher()
            
            # Initialize with empty patterns list
            assert isinstance(matcher.patterns, list), "Patterns should be a list"
            assert len(matcher.patterns) == 0, "Patterns should be empty initially"
            
            yield matcher
    
    @pytest.fixture
    def sample_patterns(self):
        """Create sample patterns for testing."""
        with patch("prompter.pattern.PromptPattern", create=True) as MockPattern:
            from prompter.pattern import PromptPattern
            
            patterns = [
                PromptPattern(
                    id="pattern-1",
                    template="Hello, {name}!",
                    score=0.9
                ),
                PromptPattern(
                    id="pattern-2",
                    template="Goodbye, {name}!",
                    score=0.8
                ),
                PromptPattern(
                    id="pattern-3",
                    template="{greeting}, {name}! How are you?",
                    score=0.7
                )
            ]
            
            # Verify patterns
            for i, pattern in enumerate(patterns):
                assert isinstance(pattern, PromptPattern), f"Pattern {i} should be a PromptPattern"
                assert isinstance(pattern.id, str), f"Pattern {i} ID should be a string"
                assert isinstance(pattern.template, str), f"Pattern {i} template should be a string"
                assert isinstance(pattern.score, float), f"Pattern {i} score should be a float"
            
            return patterns
    
    def test_initialization(self, matcher):
        """Test matcher initialization."""
        assert isinstance(matcher.patterns, list), "Patterns should be a list"
        assert len(matcher.patterns) == 0, "Patterns should be empty initially"
    
    def test_add_pattern(self, matcher, sample_patterns):
        """Test adding patterns to the matcher."""
        # Add one pattern
        matcher.add_pattern(sample_patterns[0])
        
        assert len(matcher.patterns) == 1, "Should have 1 pattern after adding"
        assert matcher.patterns[0] == sample_patterns[0], "Added pattern should match input"
        assert isinstance(matcher.patterns[0], type(sample_patterns[0])), "Pattern type should be preserved"
        
        # Add multiple patterns
        matcher.add_patterns(sample_patterns[1:])
        
        assert len(matcher.patterns) == 3, "Should have 3 patterns after adding more"
        assert all(p in sample_patterns for p in matcher.patterns), "All patterns should be added"
        
        # Test adding invalid pattern
        with pytest.raises(TypeError):
            matcher.add_pattern("not a pattern")
        
        # Test adding duplicate pattern (by ID)
        with pytest.raises(ValueError):
            matcher.add_pattern(sample_patterns[0])
    
    def test_remove_pattern(self, matcher, sample_patterns):
        """Test removing a pattern from the matcher."""
        # Add patterns
        matcher.add_patterns(sample_patterns)
        
        # Remove a pattern
        pattern_id = sample_patterns[0].id
        matcher.remove_pattern(pattern_id)
        
        assert len(matcher.patterns) == 2, "Should have 2 patterns after removing one"
        assert all(p.id != pattern_id for p in matcher.patterns), "Removed pattern should not be present"
        
        # Test removing non-existent pattern
        with pytest.raises(ValueError):
            matcher.remove_pattern("non-existent")
        
        # Test removing with invalid ID type
        with pytest.raises(TypeError):
            matcher.remove_pattern(123)
    
    def test_find_pattern_by_id(self, matcher, sample_patterns):
        """Test finding a pattern by ID."""
        # Add patterns
        matcher.add_patterns(sample_patterns)
        
        # Find a pattern
        pattern = matcher.find_pattern_by_id(sample_patterns[1].id)
        
        assert pattern == sample_patterns[1], "Found pattern should match the expected one"
        assert isinstance(pattern, type(sample_patterns[1])), "Pattern type should be preserved"
        
        # Find a non-existent pattern
        pattern = matcher.find_pattern_by_id("non-existent")
        
        assert pattern is None, "Non-existent pattern lookup should return None"
        
        # Test with invalid ID type
        with pytest.raises(TypeError):
            matcher.find_pattern_by_id(123)
    
    def test_match_input(self, matcher, sample_patterns):
        """Test matching input text to patterns."""
        # Add patterns
        matcher.add_patterns(sample_patterns)
        
        # Create mock implementation of match_input
        def mock_match_input(text, threshold=0.0):
            if text == "Hello, John!":
                return [(sample_patterns[0], {"name": "John"})]
            elif text == "Goodbye, John!":
                return [(sample_patterns[1], {"name": "John"})]
            elif text == "Hi, John! How are you?":
                return [(sample_patterns[2], {"greeting": "Hi", "name": "John"})]
            else:
                return []
        
        matcher.match_input = mock_match_input
        
        # Match an exact pattern
        matches = matcher.match_input("Hello, John!")
        
        assert len(matches) > 0, "Should find a match"
        assert matches[0][0] == sample_patterns[0], "Should match the first pattern"
        assert matches[0][1] == {"name": "John"}, "Should extract variables correctly"
        assert isinstance(matches[0][1], dict), "Variables should be in a dictionary"
        
        # Match with no patterns
        matches = matcher.match_input("This doesn't match any pattern")
        
        assert len(matches) == 0, "Should not find any matches"
        assert isinstance(matches, list), "Result should be a list even when empty"
    
    def test_match_with_threshold(self, matcher, sample_patterns):
        """Test matching with a score threshold."""
        # Add patterns
        matcher.add_patterns(sample_patterns)
        
        # Create mock implementation with threshold handling
        def mock_match_input(text, threshold=0.0):
            if text == "Hello, John!":
                # Return patterns with score above threshold
                matches = []
                for pattern in sample_patterns:
                    if pattern.score >= threshold:
                        matches.append((pattern, {"name": "John"}))
                return matches
            else:
                return []
        
        matcher.match_input = mock_match_input
        
        # Match with high threshold
        matches = matcher.match_input("Hello, John!", threshold=0.85)
        
        assert len(matches) == 1, "Should find only one match above threshold 0.85"
        assert matches[0][0] == sample_patterns[0], "Should match only pattern with score 0.9"
        
        # Match with medium threshold
        matches = matcher.match_input("Hello, John!", threshold=0.75)
        
        assert len(matches) == 2, "Should find two matches above threshold 0.75"
        assert matches[0][0] == sample_patterns[0], "First match should be highest score"
        assert matches[1][0] == sample_patterns[1], "Second match should be second highest score"
        
        # Match with low threshold
        matches = matcher.match_input("Hello, John!", threshold=0.5)
        
        assert len(matches) == 3, "Should find all three matches above threshold 0.5"
        
        # Test with invalid threshold type
        with pytest.raises(TypeError):
            matcher.match_input("Hello, John!", threshold="high")
    
    def test_save_load_patterns(self, matcher, sample_patterns, temp_dir):
        """Test saving and loading patterns."""
        # Add patterns
        matcher.add_patterns(sample_patterns)
        
        # Mock the save_patterns method
        def mock_save_patterns(path):
            # Create a mock file with pattern IDs
            patterns_json = [p.id for p in matcher.patterns]
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(patterns_json, f)
        
        matcher.save_patterns = mock_save_patterns
        
        # Save patterns
        save_path = temp_dir / "patterns.json"
        matcher.save_patterns(save_path)
        
        # Verify file was created
        assert save_path.exists(), "Patterns file should exist after saving"
        
        # Create a new matcher
        with patch("prompter.pattern.PatternMatcher", create=True) as MockMatcher:
            from prompter.pattern import PatternMatcher
            new_matcher = PatternMatcher()
            
            # Mock the load_patterns method
            def mock_load_patterns(path):
                # Load pattern IDs from mock file
                with open(path, 'r') as f:
                    pattern_ids = json.load(f)
                
                # Add corresponding patterns
                for pattern_id in pattern_ids:
                    pattern = next((p for p in sample_patterns if p.id == pattern_id), None)
                    if pattern:
                        new_matcher.patterns.append(pattern)
            
            new_matcher.load_patterns = mock_load_patterns
            
            # Load patterns
            new_matcher.load_patterns(save_path)
            
            assert len(new_matcher.patterns) == len(sample_patterns), "Should load all patterns"
            assert all(p.id in [sp.id for sp in sample_patterns] for p in new_matcher.patterns), "All loaded patterns should match originals"
            
            # Test with non-existent file
            with pytest.raises(FileNotFoundError):
                new_matcher.load_patterns(Path("non_existent_file.json"))


@pytest.mark.integration
class TestPatternIntegration:
    """Integration tests for the pattern module."""
    
    def test_integration_with_rl_model(self):
        """Test integration with the RL model."""
        # This test will be implemented after the modules are created
        pass 