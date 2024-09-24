from random import choice
from enum import Enum



class SafeAIStrEnum(str, Enum):
    """Generic Enum class for Safe AI Inputs"""
    def describe(self) -> str:
        """_summary_"""
        return f"This is a Safe AI Enum for {self.__class__.__name__} with value {self.name}"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"
    
    @classmethod
    def default(cls) -> str:
        """Returns a default value for the Enum"""
        return choice([i.value for i in cls])
    
    @classmethod
    def to_list(cls) -> list[str]:
        """Returns a list of string formatted Enum values"""
        return [i.value for i in cls]


class Country(SafeAIStrEnum):
    """Sample Countries for Agent Context"""
    UNITED_STATES = "United States"
    UNITED_KINGDOM = "United Kingdom"
    CANADA = "Canada"
    AUSTRALIA = "Australia"
    GLOBAL = "Global"
    
class FileExtension(SafeAIStrEnum):
    """Accepted formats for data sample upload"""
    CSV = "csv"
    #txt = "txt"


class AgentContext(SafeAIStrEnum):
    """Agent Task"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

class AgentDomain(SafeAIStrEnum):
    """Sample Business Sectors for Agent Context"""
    AGRICULTURE = 'agriculture'
    BANKING = 'banking'
    EDUCATION = 'education'
    ENERGY = 'energy'
    HEALTHCARE = 'healthcare'
    INSURANCE = 'insurance'
    FINANCE = 'finance'


class AgentModel(SafeAIStrEnum):
    """LLM Models to evaluate"""
    GPT4 = "gpt4"
    LLAMA3 = "llama3"
    PALM = "palm"
    Anthropic = "Anthropic"

class AgentTask(SafeAIStrEnum):
    """_summary_"""
    PREDICTION = "Prediction"
    DATA_GENERATION = "Text Generation"
    #SENTIMENT_ANALYSIS = "Sentiment Analysis"
    #TEXT_CLASSIFICATION = "Text Classification"
    #
    