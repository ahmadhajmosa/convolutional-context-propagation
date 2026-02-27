"""
Backward-compatible imports for CCP.

New structure:
- ccp.module: CPP
- ccp.judge: LLMJudge
- ccp.runner: main
"""

from .judge import LLMJudge
from .module import CPP, CCPPipeline
from .runner import main

__all__ = ["CPP", "CCPPipeline", "LLMJudge", "main"]
