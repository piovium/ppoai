from __future__ import annotations

from .schema import DeckSpec, Matchup

SAMPLE_DECK_A = DeckSpec(
    name="sample_a",
    characters=(1411, 1510, 2103),
    cards=(
        214111,
        214111,
        215101,
        311503,
        312004,
        312004,
        312025,
        312025,
        312029,
        312029,
        321002,
        321011,
        321016,
        321016,
        322002,
        322009,
        322009,
        330008,
        332002,
        332002,
        332004,
        332004,
        332005,
        332005,
        332006,
        332006,
        332018,
        332025,
        333004,
        333004,
    ),
)

SAMPLE_DECK_B = DeckSpec(
    name="sample_b",
    characters=(1609, 2203, 1608),
    cards=(
        216091,
        216091,
        222031,
        312004,
        312004,
        312021,
        312021,
        312025,
        312025,
        321002,
        321002,
        321011,
        322025,
        323004,
        323004,
        330005,
        331601,
        331601,
        332002,
        332003,
        332003,
        332004,
        332004,
        332005,
        332005,
        332006,
        332025,
        332025,
        333003,
        333003,
    ),
)

SMALL_DECK_POOL = (SAMPLE_DECK_A, SAMPLE_DECK_B)
SMALL_DECK_MATCHUPS = (
    Matchup("sample_a", "sample_a"),
    Matchup("sample_a", "sample_b"),
    Matchup("sample_b", "sample_a"),
    Matchup("sample_b", "sample_b"),
)


def deck_by_name(name: str) -> DeckSpec:
    for deck in SMALL_DECK_POOL:
        if deck.name == name:
            return deck
    raise KeyError(f"unknown deck: {name}")


def initial_card_vocabulary() -> tuple[int, ...]:
    ids = set[int]()
    for deck in SMALL_DECK_POOL:
        ids.update(deck.cards)
    return tuple(sorted(ids))


def initial_character_vocabulary() -> tuple[int, ...]:
    ids = set[int]()
    for deck in SMALL_DECK_POOL:
        ids.update(deck.characters)
    return tuple(sorted(ids))
