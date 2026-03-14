from __future__ import annotations

from pydantic import BaseModel, Field


class HoldingInput(BaseModel):
    symbol: str
    quantity: int = Field(ge=0)
    sellable_quantity: int | None = Field(default=None, ge=0)
    avg_cost: float = Field(ge=0.0)
    buy_date: str = ""


class TrainRequest(BaseModel):
    refresh_data: bool = False


class DecisionRequest(BaseModel):
    cash: float = Field(ge=0.0)
    refresh_data: bool = True
    holdings: list[HoldingInput] = Field(default_factory=list)
