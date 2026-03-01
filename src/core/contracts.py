from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass(frozen=True)
class AnnotationSample:
    sample_id: str
    text: str


@dataclass(frozen=True)
class AnnotationModel:
    provider: str
    name: str


class AnnotationProvider(Protocol):
    def annotate(self, sample: AnnotationSample) -> Dict[str, Any]:
        ...
