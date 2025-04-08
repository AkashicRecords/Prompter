import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from pathlib import Path
import json
import torch
from typing import Optional, List
from pydantic import BaseModel
import numpy as np
from datetime import datetime
from .rl_model import PromptRL

app = typer.Typer()
console = Console()

class PromptPattern(BaseModel):
    phrase: str
    context: str
    frequency: int = 0
    last_used: Optional[datetime] = None
    flags: List[str] = []
    success_rate: float = 0.0

class PromptEngineer:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.patterns_file = data_dir / "patterns.json"
        self.patterns = self._load_patterns()
        self.rl_model = PromptRL()
        
    def _load_patterns(self) -> List[PromptPattern]:
        if self.patterns_file.exists():
            with open(self.patterns_file) as f:
                data = json.load(f)
                return [PromptPattern(**pattern) for pattern in data]
        return []
    
    def _save_patterns(self):
        with open(self.patterns_file, "w") as f:
            json.dump([pattern.dict() for pattern in self.patterns], f, indent=2)
    
    def add_pattern(self, phrase: str, context: str, flags: List[str]):
        pattern = PromptPattern(
            phrase=phrase,
            context=context,
            flags=flags,
            last_used=datetime.now()
        )
        self.patterns.append(pattern)
        self._save_patterns()
        console.print(f"[green]Added new pattern: {phrase}[/green]")
    
    def list_patterns(self):
        for pattern in self.patterns:
            console.print(Panel(
                f"[bold]{pattern.phrase}[/bold]\n"
                f"Context: {pattern.context}\n"
                f"Frequency: {pattern.frequency}\n"
                f"Flags: {', '.join(pattern.flags)}\n"
                f"Success Rate: {pattern.success_rate:.2%}",
                title="Pattern"
            ))
    
    def analyze_patterns(self):
        if not self.patterns:
            console.print("[yellow]No patterns to analyze[/yellow]")
            return
        
        total_patterns = len(self.patterns)
        total_usage = sum(p.frequency for p in self.patterns)
        avg_success = np.mean([p.success_rate for p in self.patterns])
        
        console.print(Panel(
            f"Total Patterns: {total_patterns}\n"
            f"Total Usage: {total_usage}\n"
            f"Average Success Rate: {avg_success:.2%}",
            title="Analysis Results"
        ))
    
    def train_model(self):
        if not self.patterns:
            console.print("[yellow]No patterns to train on[/yellow]")
            return
        
        console.print("[yellow]Training model on patterns...[/yellow]")
        for pattern in self.patterns:
            # Simple reward function based on success rate
            reward = pattern.success_rate
            self.rl_model.add_to_memory(pattern, reward)
        
        loss = self.rl_model.train_step()
        if loss is not None:
            console.print(f"[green]Model trained successfully. Loss: {loss:.4f}[/green]")
        else:
            console.print("[yellow]Not enough data to train model[/yellow]")
    
    def suggest_improvements(self, pattern: PromptPattern):
        suggestions = self.rl_model.suggest_improvements(pattern)
        if suggestions:
            console.print(Panel(
                "\n".join(f"- {s}" for s in suggestions),
                title="Suggested Improvements"
            ))
        else:
            console.print("[yellow]No suggestions available for this pattern[/yellow]")

@app.command()
def add(
    phrase: str = typer.Option(..., prompt="Enter the phrase to standardize"),
    context: str = typer.Option(..., prompt="Enter the context for this phrase"),
    flags: str = typer.Option("", prompt="Enter any flags (comma-separated)")
):
    """Add a new prompt pattern"""
    engineer = PromptEngineer(Path.home() / ".prompter")
    flag_list = [f.strip() for f in flags.split(",") if f.strip()]
    engineer.add_pattern(phrase, context, flag_list)

@app.command()
def list():
    """List all saved patterns"""
    engineer = PromptEngineer(Path.home() / ".prompter")
    engineer.list_patterns()

@app.command()
def analyze():
    """Analyze your prompt patterns"""
    engineer = PromptEngineer(Path.home() / ".prompter")
    engineer.analyze_patterns()

@app.command()
def train():
    """Train the model on your patterns"""
    engineer = PromptEngineer(Path.home() / ".prompter")
    engineer.train_model()

@app.command()
def suggest(
    phrase: str = typer.Option(..., prompt="Enter the phrase to get suggestions for")
):
    """Get AI-powered prompt suggestions"""
    engineer = PromptEngineer(Path.home() / ".prompter")
    # Find the pattern with the given phrase
    pattern = next((p for p in engineer.patterns if p.phrase == phrase), None)
    if pattern:
        engineer.suggest_improvements(pattern)
    else:
        console.print(f"[red]Pattern not found: {phrase}[/red]")

def main():
    app() 