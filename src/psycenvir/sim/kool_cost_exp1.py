"""Recorded-path Kool et al. (2017) cost-task exp1: spaceship only."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import KoolCostExp1Day


class KoolCostExp1RecordedEnv:
    """Exact spaceship days with pooled treasure counterfactuals."""

    def __init__(
        self,
        experiment_id: str,
        days: Iterable[KoolCostExp1Day],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.experiment_id = experiment_id
        self.days: List[KoolCostExp1Day] = list(days)
        if not self.days:
            raise ValueError("KoolCostExp1RecordedEnv requires at least one day.")
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._day_idx = 0
        self._treasure_total = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        del seed
        self._day_idx = 0
        self._treasure_total = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_day(self.days[0])),
            self._info(None, None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed KoolCostExp1RecordedEnv; call reset().")
        submitted = normalize_action(action)
        day = self.days[self._day_idx]
        if submitted not in day.ships:
            self._done = True
            info = self._info(day, None)
            info["invalid_action"] = submitted
            return "Invalid spaceship action <<{}>>.".format(submitted), 0.0, False, True, info
        if submitted not in day.planet_by_ship:
            self._done = True
            info = self._info(day, None)
            info["unsupported_counterfactual"] = "unknown_ship_planet_map"
            return (
                "No pooled planet mapping is available for spaceship <<{}>>.".format(submitted),
                0.0,
                False,
                True,
                info,
            )

        if submitted == day.human_ship:
            base_treasure = day.human_base_treasure
            received = day.human_received
        else:
            base_treasure = int(round(day.pooled_treasure_by_ship.get(submitted, 0.0)))
            received = base_treasure * day.multiplier

        planet = day.planet_by_ship[submitted]
        self._treasure_total += received
        prefix = self._multiplier_prefix(day)
        feedback = (
            "{prefix}You are presented with spaceships {s0} and {s1}. You press <<{ship}>>. "
            "You end up on planet {planet}. You find {base} pieces of space treasure. "
            "You receive {received} pieces of space treasure."
        ).format(
            prefix=prefix,
            s0=day.ships[0],
            s1=day.ships[1],
            ship=submitted,
            planet=planet,
            base=base_treasure,
            received=received,
        )
        self._day_idx += 1
        self._done = self._day_idx >= len(self.days)
        return feedback, float(received), self._done, False, self._info(day, submitted)

    @staticmethod
    def _multiplier_prefix(day: KoolCostExp1Day) -> str:
        if day.multiplier > 1:
            return "There is a treasure multiplier. "
        return "There is no treasure multiplier. "

    def _render_day(self, day: KoolCostExp1Day) -> str:
        return (
            self._multiplier_prefix(day)
            + "You are presented with spaceships {} and {}.".format(day.ships[0], day.ships[1])
        )

    def _info(self, day: Optional[KoolCostExp1Day], selected_ship: Optional[str]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "day_idx": self._day_idx,
            "treasure_total": self._treasure_total,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "exact_transition",
            "instruction_shown": bool(self.instruction),
            "selected_ship": selected_ship,
        }
        if day is not None and self.include_human_ref:
            info["human_ref"] = day.human_ship
            info["multiplier"] = day.multiplier
        return info
