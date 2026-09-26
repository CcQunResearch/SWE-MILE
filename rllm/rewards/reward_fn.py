from typing import Protocol, runtime_checkable

from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput


@runtime_checkable
class RewardFunction(Protocol):
    """Protocol for reward functions"""

    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        """
        Calculate the reward for an agent's action.

        Args:
            task_info: The task dictionary containing question, answer, and other metadata
            action: The agent's response/solution

        Returns:
            RewardOutput: The calculated reward value, either as a float or RewardOutput object
        """
        ...


# Simple example implementation
def zero_reward(task_info: dict, action: str) -> RewardOutput:
    """
    A simple reward function that always returns zero.
    Useful as a placeholder when no specific reward logic is needed.

    Args:
        task: The task dictionary
        action: The agent's response

    Returns:
        float: Always returns 0.0
    """
    return RewardOutput(reward=0.0, metadata={})








