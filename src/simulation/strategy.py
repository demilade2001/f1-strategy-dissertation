from itertools import combinations, product
from math import comb
from typing import Iterable, List, Optional, Sequence, Tuple

from .config import MIN_STINT_LENGTH_LAPS


StrategyStop = Tuple[int, str]


def _distinct_dry_compounds_used(
    starting_compound: str,
    stops: Sequence[StrategyStop],
    dry_compounds: Sequence[str],
) -> int:
    dry_set = set(dry_compounds)
    used = []
    if starting_compound in dry_set:
        used.append(starting_compound)
    used.extend(compound for _, compound in stops if compound in dry_set)
    return len(set(used))


def is_strategy_feasible(
    race_length: int,
    starting_compound: str,
    is_wet_race: bool,
    stops: Sequence[StrategyStop],
    dry_compounds: Sequence[str],
    max_stops: int,
    min_stint_length_laps: int = MIN_STINT_LENGTH_LAPS,
) -> bool:
    stop_count = len(stops)
    if stop_count < 1 or stop_count > max_stops:
        return False

    prev_lap = None
    dry_set = set(dry_compounds)
    for pit_lap, compound in stops:
        if pit_lap < 1 or pit_lap > race_length - 1:
            return False
        if prev_lap is not None and pit_lap <= prev_lap:
            return False
        if compound not in dry_set:
            return False
        prev_lap = pit_lap

    stint_boundaries = [0] + [int(pit_lap) for pit_lap, _ in stops] + [int(race_length)]
    stint_lengths = [stint_boundaries[i + 1] - stint_boundaries[i] for i in range(len(stint_boundaries) - 1)]
    if any(length < int(min_stint_length_laps) for length in stint_lengths):
        return False

    if not is_wet_race and _distinct_dry_compounds_used(starting_compound, stops, dry_compounds) < 2:
        return False

    return True


def strategy_to_key(starting_compound: str, stops: Sequence[StrategyStop]) -> Tuple[str, Tuple[StrategyStop, ...]]:
    return starting_compound, tuple((int(lap), str(compound)) for lap, compound in stops)


class FeasibleStrategySpace(list):
    def __init__(
        self,
        race_length: int,
        starting_compound: str,
        is_wet_race: bool,
        max_stops: int,
        min_stint_length_laps: int = MIN_STINT_LENGTH_LAPS,
        dry_compounds: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.race_length = int(race_length)
        self.starting_compound = str(starting_compound)
        self.is_wet_race = bool(is_wet_race)
        self.max_stops = int(max_stops)
        self.min_stint_length_laps = int(min_stint_length_laps)
        self.dry_compounds = list(dry_compounds or ["SOFT", "MEDIUM", "HARD"])
        self._size = self._count_strategies()

    def _compound_sequence_count(self, stop_count: int) -> int:
        base = len(self.dry_compounds) ** stop_count
        if self.is_wet_race:
            return base
        if self.starting_compound in set(self.dry_compounds):
            return base - 1
        return base

    def _count_strategies(self) -> int:
        total = 0
        pit_lap_choices = range(1, self.race_length)
        for stop_count in range(1, self.max_stops + 1):
            valid_pit_lap_sequences = 0
            for pit_laps in combinations(pit_lap_choices, stop_count):
                stint_boundaries = [0] + list(pit_laps) + [self.race_length]
                stint_lengths = [
                    stint_boundaries[i + 1] - stint_boundaries[i]
                    for i in range(len(stint_boundaries) - 1)
                ]
                if any(length < self.min_stint_length_laps for length in stint_lengths):
                    continue
                valid_pit_lap_sequences += 1
            total += valid_pit_lap_sequences * self._compound_sequence_count(stop_count)
        return total

    def __len__(self) -> int:
        return self._size

    def __contains__(self, item) -> bool:
        if isinstance(item, dict):
            starting_compound = item.get("starting_compound")
            stops = item.get("stops", [])
        elif isinstance(item, tuple) and len(item) == 2:
            starting_compound, stops = item
        else:
            return False

        if starting_compound != self.starting_compound:
            return False

        return is_strategy_feasible(
            race_length=self.race_length,
            starting_compound=self.starting_compound,
            is_wet_race=self.is_wet_race,
            stops=stops,
            dry_compounds=self.dry_compounds,
            max_stops=self.max_stops,
            min_stint_length_laps=self.min_stint_length_laps,
        )

    def materialize(self, limit: Optional[int] = None) -> List[dict]:
        strategies: List[dict] = []
        if limit is not None and self._size > limit:
            raise ValueError(
                f"Strategy space has {self._size} elements, above materialization limit {limit}"
            )

        for stop_count in range(1, self.max_stops + 1):
            for pit_laps in combinations(range(1, self.race_length), stop_count):
                for compounds in product(self.dry_compounds, repeat=stop_count):
                    stops = list(zip(pit_laps, compounds))
                    if not is_strategy_feasible(
                        race_length=self.race_length,
                        starting_compound=self.starting_compound,
                        is_wet_race=self.is_wet_race,
                        stops=stops,
                        dry_compounds=self.dry_compounds,
                        max_stops=self.max_stops,
                        min_stint_length_laps=self.min_stint_length_laps,
                    ):
                        continue
                    strategies.append(
                        {
                            "starting_compound": self.starting_compound,
                            "stops": [(int(lap), str(compound)) for lap, compound in stops],
                        }
                    )
        return strategies


def enumerate_feasible_strategies(
    race_length: int,
    starting_compound: str,
    is_wet_race: bool,
    max_stops: int,
    min_stint_length_laps: int = MIN_STINT_LENGTH_LAPS,
    dry_compounds: list = ["SOFT", "MEDIUM", "HARD"],
) -> list:
    return FeasibleStrategySpace(
        race_length=race_length,
        starting_compound=starting_compound,
        is_wet_race=is_wet_race,
        max_stops=max_stops,
        min_stint_length_laps=min_stint_length_laps,
        dry_compounds=dry_compounds,
    )