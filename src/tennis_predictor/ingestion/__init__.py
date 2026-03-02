"""tennis_predictor.ingestion — event-driven ingestion pipeline.

Public API:

    from tennis_predictor.ingestion import (
        MatchEvent, OddsUpdate,
        MatchEventProducer,
        EventQueue,
        GameStateManager,
        InferenceEngine,
        EventConsumer, OddsPublisher,
        IngestionConfig, IngestionPipeline,
    )
"""

from .consumer import EventConsumer, OddsPublisher
from .engine import InferenceEngine
from .models import MatchEvent, OddsUpdate
from .pipeline import IngestionConfig, IngestionPipeline
from .producer import MatchEventProducer
from .queue import EventQueue
from .state import FEATURE_COLS, GameStateManager

__all__ = [
    "MatchEvent",
    "OddsUpdate",
    "MatchEventProducer",
    "EventQueue",
    "GameStateManager",
    "FEATURE_COLS",
    "InferenceEngine",
    "EventConsumer",
    "OddsPublisher",
    "IngestionConfig",
    "IngestionPipeline",
]
