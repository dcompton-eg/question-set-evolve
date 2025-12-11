"""Prompt loading utilities."""

from pathlib import Path


def load_prompt(name: str) -> str:
    """Load a prompt file from the prompts directory."""
    prompts_dir = Path(__file__).parent
    prompt_path = prompts_dir / f"{name}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text()


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent
