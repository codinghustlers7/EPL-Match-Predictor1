import os
import csv
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

DATA_PATH = "matches (2).csv"


class EplPredictor:
    def __init__(self) -> None:
        self.latest_season: Optional[int] = None
        self.teams: List[str] = []
        # Counters
        self.league_result_counter: Counter[str] = Counter()
        self.home_pair_counter: Dict[Tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.home_team_counter: Dict[str, Counter[str]] = defaultdict(Counter)
        self.away_team_counter: Dict[str, Counter[str]] = defaultdict(Counter)

    def load_data(self, csv_path: str = DATA_PATH) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("comp") and row["comp"] != "Premier League":
                    continue
                team = row.get("team") or ""
                opponent = row.get("opponent") or ""
                venue = row.get("venue") or ""
                result = row.get("result") or ""
                season_str = row.get("season") or ""
                if not (team and opponent and venue and result and season_str):
                    continue
                if result not in {"W", "D", "L"}:
                    continue
                try:
                    season = int(float(season_str))
                except ValueError:
                    continue
                rows.append(
                    {
                        "team": team,
                        "opponent": opponent,
                        "venue": venue,
                        "result": result,
                        "season": str(season),
                    }
                )
        return rows

    def train(self, rows: List[Dict[str, str]]) -> None:
        if not rows:
            raise ValueError("No rows to train on")
        seasons = [int(r["season"]) for r in rows]
        self.latest_season = max(seasons)
        self.teams = sorted({r["team"] for r in rows})
        for r in rows:
            if r["venue"] == "Home":
                self.league_result_counter[r["result"]] += 1
                self.home_pair_counter[(r["team"], r["opponent"])][r["result"]] += 1
                self.home_team_counter[r["team"]][r["result"]] += 1
                # From the away perspective, invert result W/L
                inv = {"W": "L", "L": "W", "D": "D"}[r["result"]]
                self.away_team_counter[r["opponent"]][inv] += 1

    def get_teams(self) -> List[str]:
        return self.teams

    @staticmethod
    def _to_prob(counter: Counter[str], alpha: float = 1.0) -> Dict[str, float]:
        keys = ["W", "D", "L"]
        total = sum(counter[k] for k in keys) + alpha * len(keys)
        return {k: (counter[k] + alpha) / total for k in keys}

    @staticmethod
    def _blend(probs: List[Dict[str, float]], weights: List[float]) -> Dict[str, float]:
        agg = {"W": 0.0, "D": 0.0, "L": 0.0}
        for p, w in zip(probs, weights):
            for k in agg:
                agg[k] += p.get(k, 0.0) * w
        s = sum(agg.values())
        if s <= 0:
            return {"W": 1/3, "D": 1/3, "L": 1/3}
        return {k: v / s for k, v in agg.items()}

    def predict_match(self, home_team: str, away_team: str, season: Optional[int] = None) -> Dict[str, float]:
        if season is None:
            season = self.latest_season
        # Priors
        p_league = self._to_prob(self.league_result_counter, alpha=1.0)
        p_pair = self._to_prob(self.home_pair_counter.get((home_team, away_team), Counter()), alpha=1.0)
        p_home = self._to_prob(self.home_team_counter.get(home_team, Counter()), alpha=1.0)
        p_away = self._to_prob(self.away_team_counter.get(away_team, Counter()), alpha=1.0)
        # Blend with heuristic weights
        probs = self._blend(
            [p_league, p_pair, p_home, p_away],
            [0.2, 0.4, 0.25, 0.15],
        )
        return {
            "homeWin": probs["W"],
            "draw": probs["D"],
            "awayWin": probs["L"],
            "season": int(season) if season is not None else None,
        }


def build_and_train_predictor(csv_path: str = DATA_PATH) -> EplPredictor:
    predictor = EplPredictor()
    rows = predictor.load_data(csv_path)
    predictor.train(rows)
    return predictor