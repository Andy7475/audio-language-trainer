from pydantic import BaseModel, ConfigDict, Field


from typing import List, Literal

from src.models import BCP47Language


class ChallengeBaseModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class QandA(ChallengeBaseModel):
    question: str
    answer: str


class Scenario(ChallengeBaseModel):
    role_learner: str = Field(
        ..., description="Role for learner to play (e.g., 'coffee shop customer')"
    )
    role_teacher: str = Field(
        ..., description="Role for teacher to play (e.g., 'coffee shop staff')"
    )
    situation: str = Field(
        ..., description="Setting description (e.g., 'A coffee shop')"
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ..., description="Difficulty level (e.g., 'easy', 'medium', 'hard')"
    )
    task: str = Field(..., description="Main task to complete (e.g., 'Order a coffee')")
    find_out: List[QandA] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="Specific information to discover, together with a proposed answer (e.g., 'What is the price?', 5.00)",
    )


class Challenge(ChallengeBaseModel):
    scenarios: List[Scenario] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="List of 3 roleplay scenarios, one at each difficult",
    )


class PublishedChallenge(ChallengeBaseModel):
    public_url: str
    source_language: BCP47Language
    target_language: BCP47Language


class ChallengeRecord(ChallengeBaseModel):
    story_title_hash: str = Field(..., description="Hash of the parent story")
    challenge: Challenge = Field(..., description="Challenge data")
    published: List[PublishedChallenge]
