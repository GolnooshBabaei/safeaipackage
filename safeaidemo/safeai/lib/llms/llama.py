from safeai.lib.models import ModelGeneratorFactory, ModelPredictionFactory, Chat


class LLamaPredictor(ModelPredictionFactory):
    """_summary_"""

    def image_predictor(self, chat: Chat) -> Chat:
        """_summary_"""
        raise NotImplementedError("Method form not implemented")

    def text_predictor(self, chat: Chat) -> Chat:
        """_summary_"""
        raise NotImplementedError("Method form not implemented")

    def audio_predictor(self):
        """_summary_"""
        raise NotImplementedError("Method form not implemented")

    def video_predictor(self):
        """_summary_"""
        raise NotImplementedError("Method form not implemented")


class LLamaGenerator(ModelGeneratorFactory):
    """_summary_"""

    def image_generator(self, stream: list[Chat]) -> Chat:
        """_summary_"""
        raise NotImplementedError("Method form not implemented")

    def text_generator(self, stream: list[Chat]) -> Chat:
        """_summary_"""
        raise NotImplementedError("Method form not implemented")

    def audio_generator(self):
        """_summary_"""
        raise NotImplementedError("Method form not implemented")

    def video_generator(self):
        """_summary_"""
        raise NotImplementedError("Method form not implemented")
