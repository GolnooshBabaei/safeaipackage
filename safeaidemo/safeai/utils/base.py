from pydantic import BaseModel, Field, model_validator, computed_field
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

class Base(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """
    
    class Config:
        """_summary_"""
        arbitrary_types_allowed = True
        use_enum_values = True