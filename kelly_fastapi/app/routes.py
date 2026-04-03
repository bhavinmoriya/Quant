from fastapi import APIRouter, HTTPException
from app.utils import simulate_betting, plot_bankroll_history
from pathlib import Path
import uuid

router = APIRouter()

@router.post("/simulate")
async def simulate_kelly(initial_bankroll: float = 10000, p: float = 0.6, b: float = 1.5, n_iterations: int = 100):
    bankroll_history = simulate_betting(initial_bankroll, p, b, n_iterations)

    plot_path = plot_bankroll_history(bankroll_history, initial_bankroll, p, b, n_iterations)

    return {
        "bankroll_history": bankroll_history,
        "plot_url": f"/static/{plot_path.name}"
    }
