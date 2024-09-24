from typing import Union
from safeai.enums import AgentModel
from safeai.lib.llms.llama import LLamaGenerator, LLamaPredictor
from safeai.lib.llms.palm import PalmGenerator, PalmPredictor
from safeai.lib.llms.gpt import OpenAIGenerator, OpenAIPredictor
from safeai.lib.llms.anthropic import AnthropicGenerator, AnthropicPredictor


MG = Union[LLamaGenerator, PalmGenerator, AnthropicGenerator, OpenAIGenerator]
MP = Union[LLamaPredictor, PalmPredictor, AnthropicPredictor, OpenAIPredictor]


class GeneratorModelStrategy:
    """_summary_"""
    def __init__(self, agent_model:AgentModel) -> None:
        self.agent_model = agent_model

    def __call__(self) -> MG:
        if self.agent_model == AgentModel.LLAMA3:
            return LLamaGenerator()
        elif self.agent_model == AgentModel.PALM:
            return PalmGenerator()
        elif self.agent_model == AgentModel.Anthropic:
            return AnthropicGenerator()
        else:
            return OpenAIGenerator()
        
        
class PredictorModelStrategy:
    """_summary_"""
    def __init__(self, agent_model:AgentModel) -> None:
        self.agent_model = agent_model

    def __call__(self) -> MP:
        if self.agent_model == AgentModel.LLAMA3:
            return LLamaPredictor()
        elif self.agent_model == AgentModel.PALM:
            return PalmPredictor()
        elif self.agent_model == AgentModel.Anthropic:
            return AnthropicPredictor()
        else:
            return OpenAIPredictor()
        
class AgentModelFactory:
    """_summary_"""
    def __init__(self, agent_model:AgentModel) -> None:
        self.agent_model = agent_model
        self.agent_generator = GeneratorModelStrategy(agent_model)
        self.agent_predictor = PredictorModelStrategy(agent_model)