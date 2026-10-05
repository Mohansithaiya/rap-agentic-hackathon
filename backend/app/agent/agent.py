"""A small deterministic agent for inventory reorder questions."""

import os
import re

from openai import OpenAI
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


class RequestInterpretation(BaseModel):
    """The limited information the LLM may extract from a user request."""

    product: str
    operation: str


class InventoryAgent:
    """Interpret inventory requests and execute tools only through a harness."""

    def __init__(self, harness: Harness | None = None) -> None:
        self.harness = harness or Harness()

    @staticmethod
    def _interpret_with_llm(request: str) -> RequestInterpretation | None:
        """Extract a product and operation; never expose tools to the LLM."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract only the product name and requested operation from the user. "
                            "Return a JSON object with string fields product and operation. "
                            "Do not answer the request, make inventory decisions, or call tools. "
                            "Use an empty product if none is identifiable."
                        ),
                    },
                    {"role": "user", "content": request},
                ],
            )
            content = response.choices[0].message.content
            if not content:
                return None
            parsed = RequestInterpretation.model_validate_json(content)
            product = parsed.product.strip().strip(" \t\r\n.,?!")
            operation = parsed.operation.strip()
            if not product or not operation:
                return None
            return RequestInterpretation(product=product, operation=operation)
        except Exception:
            # Missing SDK setup, network errors, and malformed model output all
            # fall back to deterministic request parsing.
            return None

    @staticmethod
    def _extract_deterministically(request: str) -> str:
        """Use the original deterministic reorder phrasing as a fallback."""
        match = re.search(r"\breorder\s+(.+?)\s*[?.!]*$", request, re.IGNORECASE)
        if not match:
            return ""
        return match.group(1).strip().strip(" \t\r\n.,?!")

    def run(self, request: str) -> InventoryResponse:
        """Look up the requested product and determine whether to reorder."""
        interpretation = self._interpret_with_llm(request)
        product = (
            interpretation.product
            if interpretation is not None
            else self._extract_deterministically(request)
        )
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
        details_outcome = self.harness.execute(
            tools.get_product_details, product, tool_name="get_product_details"
        )
        if details_outcome.success:
            details = details_outcome.result
            explanation = (
                f"{inventory.product} has quantity {inventory.quantity}, {decision} "
                f"the reorder level of {inventory.reorder_level}; {action}. "
                f"It is a {details.category} product: {details.description} "
                f"Unit price: ${details.unit_price:.2f}."
            )
        else:
            explanation = (
                f"{inventory.product} has quantity {inventory.quantity}, {decision} "
                f"the reorder level of {inventory.reorder_level}; {action}. "
                f"Product details were unavailable: {details_outcome.error}"
            )
        return InventoryResponse(
            product=inventory.product,
            quantity=inventory.quantity,
            reorder_level=inventory.reorder_level,
            reorder_needed=reorder_needed,
            explanation=explanation,
        )
