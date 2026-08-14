"""Display-ready Pick 6 ticket rows."""

from __future__ import annotations

from src.pick6.domain import Pick6Ticket
from src.pick6.simulation import TicketSimulation


def ticket_summary(ticket: Pick6Ticket, simulation: TicketSimulation) -> dict[str, object]:
    return {
        "ticket_id": ticket.ticket_id,
        "legs": len(ticket.legs),
        "joint_probability": simulation.joint_probability,
        "payout_multiple": float(ticket.payout_multiple),
        "break_even_probability": ticket.break_even_probability,
        "expected_value_per_unit": simulation.expected_value_per_unit,
        "dependency_warning": ticket.dependency_warning,
        "stake_cap": float(ticket.stake),
    }
