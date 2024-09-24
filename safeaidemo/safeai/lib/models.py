from abc import ABC, abstractmethod
from functools import cached_property
from safeai.utils.base import Base, Field

class Chat(Base):
    """_summary_"""
    role: str = Field(
        "system",
        title="Role",
        description="The role of the chat user"
    )
    content: str = Field(
        ...,
        title="Content",
        description="The content of the chat"
    )


class ModelGeneratorFactory(ABC):
    """_summary_"""
    
    @cached_property
    @abstractmethod
    def client(self):
        """_summary_"""
        raise NotImplementedError("Method form not implemented")
    
    @abstractmethod
    def image_generator(self, stream:list[Chat]) -> Chat:
        """_summary_"""
        raise NotImplementedError("Method form not implemented")
    
    
    @abstractmethod
    def text_generator(self, stream:list[Chat]) -> Chat:
        """_summary_"""
        raise NotImplementedError("Method form not implemented")
    
    @abstractmethod
    def audio_generator(self):
        """_summary_"""
        raise NotImplementedError("Method form not implemented")
    
    @abstractmethod
    def video_generator(self):
        """_summary_"""
        raise NotImplementedError("Method form not implemented")
    

class ModelPredictionFactory(ABC):
    """_summary_"""
    
    @cached_property
    @abstractmethod
    def client(self):
        """_summary_"""
        raise NotImplementedError("Method form not implemented")
    
    @abstractmethod
    def image_predictor(self, chat:Chat) -> Chat:
        """_summary_"""
        raise NotImplementedError("Method form not implemented")
    
    
    @abstractmethod
    def text_predictor(self, chat:Chat) -> Chat:
        """_summary_"""
        raise NotImplementedError("Method form not implemented")
    
    @abstractmethod
    def audio_predictor(self):
        """_summary_"""
        raise NotImplementedError("Method form not implemented")
    
    @abstractmethod
    def video_predictor(self):
        """_summary_"""
        raise NotImplementedError("Method form not implemented")
