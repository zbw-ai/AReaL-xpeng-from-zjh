"""Config extensions for agentic training scenarios."""

from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig


@dataclass
class AgenticConfig(GRPOConfig):
    """Extended config for agentic RL scenarios (Search R1, Code DAPO)."""

    # Search R1
    retrieval_endpoint: str = ""
    max_turns: int = 10
    max_tool_uses: int = 2
    max_total_tokens: int = 12800
    system_prompt: str = "You're a helpful assistant."

    # Code DAPO
    sandbox_type: str = "local"  # "local" | "execd"
    execd_endpoint: str = ""
    code_timeout: int = 10
