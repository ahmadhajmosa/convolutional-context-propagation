"""
Backward-compatible imports for CCP.

New structure:
- ccp.module: CCPPipeline
- ccp.judge: LLMJudge
- ccp.runner: main
"""

from .judge import LLMJudge
from .module import CCPPipeline
from .runner import main

__all__ = ["CCPPipeline", "LLMJudge", "main"]

