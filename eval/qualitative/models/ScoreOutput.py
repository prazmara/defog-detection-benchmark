from pydantic import BaseModel, Field, validator
from typing import Dict



class ScoreOutput(BaseModel):
    visibility_restoration: int = Field(0, ge=0, le=5)
    visual_distortion: int = Field(0, ge=0, le=5)
    boundary_clarity: int = Field(0, ge=0, le=5)
    scene_consistency: int = Field(0, ge=0, le=5)
    object_consistency: int = Field(0, ge=0, le=5)
    perceived_detectability: int = Field(0, ge=0, le=5)
    relation_consistency: int = Field(0, ge=0, le=5)
    relation_mismatches: list = Field(default_factory=list)  # optional; only if relation_consistency < 5
    explanation: str = ""

    @property
    def total(self) -> int:
        return self.visibility_restoration + self.visual_distortion + self.boundary_clarity + self.scene_consistency + self.object_consistency + self.perceived_detectability

