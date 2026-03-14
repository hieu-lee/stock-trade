from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI

from rdtb.api.schemas import DecisionRequest, TrainRequest
from rdtb.config import get_default_config
from rdtb.service.pipeline import generate_daily_decisions, train_system


def create_app(project_dir: str | Path | None = None) -> FastAPI:
    config = get_default_config(project_dir)
    app = FastAPI(title="Really Dope Trading Bot", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "project_dir": str(config.project_dir)}

    @app.post("/train")
    def train_endpoint(payload: TrainRequest) -> dict[str, object]:
        return train_system(config=config, refresh_data=payload.refresh_data)

    @app.post("/decisions")
    def decision_endpoint(payload: DecisionRequest) -> dict[str, object]:
        holdings = pd.DataFrame([item.model_dump() for item in payload.holdings])
        return generate_daily_decisions(
            cash=payload.cash,
            holdings=holdings,
            config=config,
            refresh_data=payload.refresh_data,
        )

    return app
