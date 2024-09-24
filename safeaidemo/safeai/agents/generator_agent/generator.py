from pandas import DataFrame
from safeai.agents.generator_agent.models.prompts import PromptStrategy, PromptFactory
from safeai.enums import AgentModel
from safeai.lib.factory import GeneratorModelStrategy



    
class SyntheticTabularGenerator:
    """_summary_"""
    def __init__(self, dataframe:DataFrame, state:dict) -> None:
        self.dataframe = dataframe
        self.state = state
        self.prompt_factory = PromptFactory(state=state)
        self.model_strategy = GeneratorModelStrategy(AgentModel(self.state.get("agent_model")))
        
        
        
    def generate(self):
        stream = [
            self.prompt_factory.company_prompt(),
            self.prompt_factory.data_profile_prompt()
        ]
        for index, rows in self.dataframe.iterrows():
            c_name, c_type, c_desc = rows
            prompt_strategy = PromptStrategy(c_name, c_desc, c_type, self.state)
            stream.append(prompt_strategy().role_prompt())
        stream.append(self.prompt_factory.after_thought())
        return self.model_strategy().text_generator(stream)
