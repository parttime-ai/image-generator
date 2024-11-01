# Description: This file contains the configuration settings for the application.
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfiguration(BaseSettings):
    together_api_key: str = Field(alias="TOGETHER_API_KEY")
    model: str = Field(alias="MODEL")
    sd_model: str = Field(alias="SD_MODEL")
    is_local: bool = False
    model_config = SettingsConfigDict(env_file="config/.env")
