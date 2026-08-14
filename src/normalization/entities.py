"""Bitemporal provider-to-canonical entity resolution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.contracts.domain import stable_id


@dataclass(frozen=True)
class ProviderEntity:
    entity_type: str
    provider: str
    provider_entity_id: str
    canonical_entity_id: str
    valid_from: datetime
    valid_to: datetime | None = None

    def valid_at(self, as_of: datetime) -> bool:
        return self.valid_from <= as_of and (self.valid_to is None or as_of < self.valid_to)


class EntityResolver:
    def __init__(self, mappings: list[ProviderEntity]) -> None:
        self._mappings = tuple(mappings)

    def resolve(
        self,
        entity_type: str,
        provider: str,
        provider_entity_id: str,
        *,
        as_of: datetime,
    ) -> str:
        matches = [
            row
            for row in self._mappings
            if row.entity_type == entity_type
            and row.provider == provider
            and row.provider_entity_id == provider_entity_id
            and row.valid_at(as_of)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"Expected one {entity_type} mapping for {provider}:{provider_entity_id} at {as_of}; "
                f"found {len(matches)}"
            )
        return matches[0].canonical_entity_id


def canonical_entity_id(entity_type: str, *identity_parts: object) -> str:
    return stable_id(entity_type, *identity_parts)
