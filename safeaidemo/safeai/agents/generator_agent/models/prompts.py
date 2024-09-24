from typing import Any, override
from safeai.agents.generator_agent.models.columns import  ColumnDataType, ColumnDescriptorStrategy
from safeai.lib.models import Chat
from safeai.models import CompanyProfile, DataProfile


class PromptFactory:
    """_summary_"""
    
    def __init__(self, state:dict) -> None:
        """_summary_"""
        self.company_profile = CompanyProfile(**state)
        self.data_profile = DataProfile(**state)
        
    def company_prompt(self) -> Chat:
        """_summary_"""
        prompt = """
            You are a synthetic data generator agent for a company that wants to generate\
            synthetic data for its business. Employees of the company use the company's synthetic \
            data generator agent to generate real world data for their analytics and business teams.\
            Your job is to generate realistic real world data that can be used to train machine \
            learning models or perform realistic data analysis tasks.The datasets you generate \
            should be representative of the real world that the company operates in.\n
        """

        if self.company_profile.company_name:
            prompt += f"""The name of the company is:\n
                ```{self.company_profile.company_name}```\n"""
            
        if self.company_profile.company_description:
            prompt += f"""This is a brief description of what the company does in the real world:\n
                ```{self.company_profile.company_description}```\n"""
                
        if self.company_profile.company_business_sector:
            prompt += f"""The company offers their services in the following business sectors:\n
                ```{'\n'.join(list(self.company_profile.company_business_sector))}```\n"""
        
        if self.company_profile.company_country:
            prompt += f"""Your company operates its business in the following countries:\n \
                ```{self.company_profile.company_country}```\n"""
        return Chat(role="system", content=prompt)

        
    def example_prompt(self) -> Chat:
        """_summary_"""
        prompt = f"""
            A data point is a single unit of information that represents\
            an observation or measurement of a real word attribute or variable.\n
            A set of data points collected through mulitple observations of a variable\
            is called a feature.\n
            
            A collection of features is called a tabular dataset. In a tabular dataset,\
            data points corresponds to a row, with each value in the row representing the\
            data for a specific feature. This is an example of a tabular dataset:\n

            | Name | Age | Country | Gender | Height | Weight | Username |
            |------|-----|---------|--------|--------|--------|----------|
            | John | 25  | USA     | Male   | 6'2    | 180    | john123  |
            | Jane | 30  | UK      | Female | 5'8    | 140    | jane123  |
            | Jack | 22  | Canada  | Male   | 5'10   | 160    | jack123  |
            \n\n
            
            Each feature corresponds to a column in the dataset and can be anyone of the\
            following data types:\n
                ```{'\t-\n'.join(ColumnDataType.to_list())}```\n
                
             will provide a description for each feature they want to generate.\n\n
        """
        return Chat(role="system", content=prompt)
    
    def data_profile_prompt(self) -> Chat:
        """_summary_"""
    
        prompt = f"""
                I am a data scientist at ```{self.company_profile.company_name}```.\
                    This is a description of the data I want to generate:\n
                    ```{self.data_profile.data_sample_description}```\n"""
                    
        if self.data_profile.data_num_rows:
            prompt += f"""I want to generate ```{self.data_profile.data_num_rows}``` observations for\
                each feature in the dataset.\n"""
                
        prompt += """I will now describe the features to be generate."""
        
        return Chat(role="user", content=prompt)
    
    def after_thought(self) -> Chat:
        """_summary_"""
        prompt = """
            Return your results in flat json format. Each feature\
            should be a key in the json object with the value being a list of\
            the data points for that feature. The data points should be in the\
            same order as the number of observations you were asked to generate.\n
        """
        return Chat(role="system", content=prompt)

    
    


class DataTypePromptFactory:
    """_summary_"""
    def __init__(self, col_name:str, col_desc:str, state:dict) -> None:
        self.data_description = ColumnDescriptorStrategy(col_name, col_desc, ColumnDataType.TEXT, state)
        self.data = self.data_description().data
        self.data_profile = DataProfile(**state)
        
    def role_prompt(self) -> Chat:
        """_summary_"""
        raise NotImplementedError("Method create_prompt not implemented")
    


class TextPrompt(DataTypePromptFactory):
    """_summary_"""

    @override
    def role_prompt(self) -> Chat:
        """_summary_"""

        prompt = f"""
            I want to create a new feature called ```{self.data_description.c_name}```\
                with the following feature description:\n
                    ```{self.data_description.c_desc}```\n
                    
            Using my company's profile and data description generate ````{self.data_profile.data_num_rows}```\
            realistic real world ```{self.data.text_type}``` ```{self.data_description.c_type}``` samples for\
                the ```{self.data_description.c_name}``` feature.\n       
        """
        
        if self.data.emotions:
            prompt += f"""The ```{self.data.text_type}```s should contain text\
                with all of the following emotions:\n
                ```{'\n'.join(list(self.data.emotions))}```\n"""
                
        return Chat(role="user", content=prompt)
 




class NumericPrompt(PromptFactory):
    ...
    
class CategoricalPrompt(PromptFactory):
    ...
    
class DatePrompt(PromptFactory):
    ...
    
class TimestampPrompt(DataTypePromptFactory):
        
    @override
    def role_prompt(self) -> Chat:
        prompt = f"""
            I want to generate a new feature called ```{self.data_description.c_name}```\
                with the following feature description:\n
                    ```{self.data_description.c_desc}```\n
                    
            Using your client's company profile and data description, your job is to \
            generate ````{self.data.data_num_rows}``` realistic real world\
            ```{self.data_description.c_type}``` samples between ```{self.data.start_date}```\
                and ```{self.data.end_date}``` for the ```{self.data_description.c_name}``` feature.\n    
        """
        return Chat(role="user", content=prompt)
    
class BooleanPrompt(PromptFactory):
    ...
    
class OrdinalPrompt(PromptFactory):
    ...
    

class PromptStrategy:
    """_summary_"""
    def __init__(self, c_name:str, c_desc:str, c_type:ColumnDataType, state:dict) -> None:
        self.c_name = c_name
        self.c_desc = c_desc
        self.c_type = c_type
        self.state=state
        
    def __call__(self, *args: Any, **kwds: Any) -> DataTypePromptFactory:
        if self.c_type == ColumnDataType.TEXT:
            return TextPrompt(self.c_name, self.c_desc, self.state)
        elif self.c_type == ColumnDataType.INTEGER:
            return NumericPrompt(*args, **kwds)
        elif self.c_type == ColumnDataType.CATEGORICAL:
            return CategoricalPrompt(*args, **kwds)
        elif self.c_type == ColumnDataType.DATE:
            return DatePrompt(*args, **kwds)
        elif self.c_type == ColumnDataType.BOOLEAN:
            return BooleanPrompt(*args, **kwds)
        elif self.c_type == ColumnDataType.ORDINAL:
            return OrdinalPrompt(*args, **kwds)
        elif self.c_type == ColumnDataType.TIMESTAMP:
            return TimestampPrompt(self.c_name, self.c_desc, self.state)
        else:
            raise ValueError("Invalid column data type")