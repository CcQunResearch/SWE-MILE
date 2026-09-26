"""
Type alias for TokenOutput and TokenInput -- need to take different backends into account.
"""

from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.image_processing_utils import BaseImageProcessor

    Tokenizer: TypeAlias = PreTrainedTokenizer
    Processor: TypeAlias = ProcessorMixin
    ImageProcessor: TypeAlias = BaseImageProcessor
else:
    # make it importable from other files as a type in runtime
    Tokenizer: TypeAlias = Any
    Processor: TypeAlias = Any
    ImageProcessor: TypeAlias = Any

# Verl types
VerlTokenInput: TypeAlias = list[int]
try:
    from verl.workers.rollout.replica import TokenOutput

    VerlTokenOutput: TypeAlias = TokenOutput
except ImportError:  # avoid cases when the verl backend is not used
    VerlTokenOutput: TypeAlias = Any

# Union everything together
TokenInput: TypeAlias = VerlTokenInput
TokenOutput: TypeAlias = VerlTokenOutput
