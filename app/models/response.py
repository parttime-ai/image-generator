from enum import Enum

from pydantic import BaseModel


class SummaryOfFindings(BaseModel):
    NSFW_Content: bool
    Racism: bool
    Child_Exploitation: bool
    Pornographic_Content: bool
    Nudity: bool
    Profanity: bool
    Violence_Death: bool
    Weapons: bool


class ConfidenceLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class OverallAssessment(str, Enum):
    appropriate = "appropriate"
    inappropriate = "inappropriate"


class ContentAssessment(BaseModel):
    summary_of_findings: SummaryOfFindings
    overall_assessment: OverallAssessment
    confidence_level: ConfidenceLevel
    reason: str




