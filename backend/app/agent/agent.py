"""A small deterministic agent for inventory reorder questions."""

import re

from pydantic import BaseModel

from . import tools
from .harness import Harness


class InventoryResponse(BaseModel):
    """Structured answer to an inventory reorder question."""

    product: str
    quantity: int | None
    reorder_level: int | None
    reorder_needed: bool | None
    explanation: str


class InventoryAgent:
    """Answer inventory questions using tools only through a harness."""

    def __init__(self, harness: Harness | None = None) -> None:
        self.harness = harness or Harness()

    def run(self, request: str) -> InventoryResponse:
        """Look up the requested product and determine whether to reorder."""
        match = re.search(r"\breorder\s+(.+?)\s*[?.!]*$", request, re.IGNORECASE)
        if not match:
            return InventoryResponse(
                product="",
                quantity=None,
                reorder_level=None,
                reorder_needed=None,
                explanation="I couldn't identify a product in the request.",
            )

        product = match.group(1).strip().strip(" \t\r\n.,?!")
        if not product:
            return InventoryResponse(
                product="",
                quantity=None,
                reorder_level=None,
                reorder_needed=None,
                explanation="I couldn't identify a product in the request.",
            )

        # All tool execution is routed through the harness for limits and tracing.
        outcome = self.harness.execute(
            tools.get_inventory, product, tool_name="get_inventory"
        )
        if not outcome.success:
            return InventoryResponse(
                product=product,
                quantity=None,
                reorder_level=None,
                reorder_needed=None,
                explanation=f"Inventory is unavailable for {product}: {outcome.error}",
            )

        inventory = outcome.result
        reorder_needed = inventory.quantity <= inventory.reorder_level
        decision = "at or below" if reorder_needed else "above"
        action = "reorder is needed" if reorder_needed else "reorder is not needed"
        return InventoryResponse(
            product=inventory.product,
            quantity=inventory.quantity,
            reorder_level=inventory.reorder_level,
            reorder_needed=reorder_needed,
            explanation=(
                f"{inventory.product} has quantity {inventory.quantity}, {decision} "
                f"the reorder level of {inventory.reorder_level}; {action}."
            ),
        )
