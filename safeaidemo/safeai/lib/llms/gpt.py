from typing import Iterable
from functools import cached_property
from openai import OpenAI
from safeai.lib.models import (
    ModelGeneratorFactory,
    ModelPredictionFactory,
    Chat
)

class OpenAIGenerator(ModelGeneratorFactory):
    """_summary_"""
    
    @cached_property
    def client(self):
        """_summary_"""

        return OpenAI()
        

    
    def image_generator(self, stream:Iterable[Chat]) -> Chat:
        """_summary_"""
        raise NotImplementedError("Method form not implemented")

    def text_generator(self, stream:Iterable[Chat]) -> Chat:
        """_summary_"""
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[i.model_dump() for i in stream]
        )
        return Chat(role="assistant", content=completion.choices[0].message.content)
    
    def audio_generator(self):
        """_summary_"""
        raise NotImplementedError("Method form not implemented")
    
    def video_generator(self):
        """_summary_"""
        raise NotImplementedError("Method form not implemented")
    
    
class OpenAIPredictor(ModelPredictionFactory):
    """_summary_"""
    
    def image_predictor(self, chat:Chat) -> Chat:
        """_summary_"""
        raise NotImplementedError("Method form not implemented")
    
    
    def text_predictor(self, chat:Chat) -> Chat:
        """_summary_"""
        raise NotImplementedError("Method form not implemented")
    
    def audio_predictor(self):
        """_summary_"""
        raise NotImplementedError("Method form not implemented")
    
    def video_predictor(self):
        """_summary_"""
        raise NotImplementedError("Method form not implemented")
