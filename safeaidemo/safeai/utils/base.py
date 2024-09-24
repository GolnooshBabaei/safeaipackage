from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Env(BaseSettings):
    """_summary_"""

    openai_api_key: str
    # palm_api_key: str
    # anthropic_api_key: str
    # llama_api_key: str


class Base(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """

    class Config:
        """_summary_"""

        arbitrary_types_allowed = True
        use_enum_values = True


env = Env()
