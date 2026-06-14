"""Recorded-path Kool et al. (2017) cost-task exp2: spaceship then alien."""

from typing import Any, Dict, Iterable, List, Optional, Tuple

from psycenvir.core.base import normalize_action, render_initial_observation
from psycenvir.models import KoolCostExp2Day


class KoolCostExp2RecordedEnv:
    """Exact two-step days with pooled alien treasure counterfactuals."""

    def __init__(
        self,
        experiment_id: str,
        days: Iterable[KoolCostExp2Day],
        include_human_ref: bool = False,
        instruction: str = "",
    ) -> None:
        self.experiment_id = experiment_id
        self.days: List[KoolCostExp2Day] = list(days)
        if not self.days:
            raise ValueError("KoolCostExp2RecordedEnv requires at least one day.")
        self.instruction = instruction
        self.include_human_ref = include_human_ref
        self._day_idx = 0
        self._awaiting_alien = False
        self._selected_ship: Optional[str] = None
        self._selected_planet: Optional[str] = None
        self._treasure_total = 0.0
        self._done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        del seed
        self._day_idx = 0
        self._awaiting_alien = False
        self._selected_ship = None
        self._selected_planet = None
        self._treasure_total = 0.0
        self._done = False
        return (
            render_initial_observation(self.instruction, self._render_spaceship_prompt(self.days[0])),
            self._info(None),
        )

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Cannot step a completed KoolCostExp2RecordedEnv; call reset().")
        submitted = normalize_action(action)
        day = self.days[self._day_idx]

        if not self._awaiting_alien:
            if submitted not in day.ships:
                self._done = True
                info = self._info(day)
                info["invalid_action"] = submitted
                return "Invalid spaceship action <<{}>>.".format(submitted), 0.0, False, True, info
            if submitted not in day.planet_by_ship:
                self._done = True
                info = self._info(day)
                info["unsupported_counterfactual"] = "unknown_ship_planet_map"
                return (
                    "No pooled planet mapping is available for spaceship <<{}>>.".format(submitted),
                    0.0,
                    False,
                    True,
                    info,
                )
            self._selected_ship = submitted
            if submitted == day.human_ship:
                self._selected_planet = day.human_planet
            else:
                self._selected_planet = day.planet_by_ship[submitted]
            self._awaiting_alien = True
            prefix = self._multiplier_prefix(day)
            feedback = (
                "{prefix}You are presented with spaceships {s0} and {s1}. You press <<{ship}>>. "
                "You end up on planet {planet}. You see alien {a0} and alien {a1}."
            ).format(
                prefix=prefix,
                s0=day.ships[0],
                s1=day.ships[1],
                ship=submitted,
                planet=self._selected_planet,
                a0=day.aliens_by_planet[self._selected_planet][0],
                a1=day.aliens_by_planet[self._selected_planet][1],
            )
            return feedback, 0.0, False, False, self._info(day)

        aliens = day.aliens_by_planet[self._selected_planet]
        if submitted not in aliens:
            self._done = True
            info = self._info(day)
            info["invalid_action"] = submitted
            return "Invalid alien action <<{}>>.".format(submitted), 0.0, False, True, info

        if (
            submitted == day.human_alien
            and self._selected_ship == day.human_ship
            and self._selected_planet == day.human_planet
        ):
            base_treasure = float(day.human_base_treasure)
        else:
            base_treasure = day.pooled_treasure_by_alien[submitted]
        received = base_treasure * day.multiplier
        self._treasure_total += received
        prefix = self._multiplier_prefix(day)
        base_display = int(base_treasure) if base_treasure == int(base_treasure) else base_treasure
        feedback = (
            "{prefix}You are presented with spaceships {s0} and {s1}. You press <<{ship}>>. "
            "You end up on planet {planet}. You see alien {a0} and alien {a1}. "
            "You press <<{alien}>>. You find {base} pieces of space treasure."
        ).format(
            prefix=prefix,
            s0=day.ships[0],
            s1=day.ships[1],
            ship=self._selected_ship,
            planet=self._selected_planet,
            a0=aliens[0],
            a1=aliens[1],
            alien=submitted,
            base=base_display,
        )
        self._advance_day()
        return feedback, float(received), self._done, False, self._info(day)

    def _advance_day(self) -> None:
        self._day_idx += 1
        self._done = self._day_idx >= len(self.days)
        self._awaiting_alien = False
        self._selected_ship = None
        self._selected_planet = None

    @staticmethod
    def _multiplier_prefix(day: KoolCostExp2Day) -> str:
        if day.multiplier > 1:
            return "There is a treasure multiplier. "
        return "There is no treasure multiplier. "

    def _render_spaceship_prompt(self, day: KoolCostExp2Day) -> str:
        return (
            self._multiplier_prefix(day)
            + "You are presented with spaceships {} and {}.".format(day.ships[0], day.ships[1])
        )

    def _info(self, day: Optional[KoolCostExp2Day]) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "day_idx": self._day_idx,
            "awaiting_alien": self._awaiting_alien,
            "treasure_total": self._treasure_total,
            "feedback_causal": True,
            "reward_defined": True,
            "fidelity_level": "exact_transition",
            "instruction_shown": bool(self.instruction),
        }
        if day is not None and self.include_human_ref:
            info["human_ship"] = day.human_ship
            info["human_alien"] = day.human_alien
        return info
