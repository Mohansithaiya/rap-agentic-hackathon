"""Practice tools backed by deterministic mock data."""

from pydantic import BaseModel


class InventoryResult(BaseModel):
    """Inventory details for a product."""

    product: str
    quantity: int
    reorder_level: int


_INVENTORY: dict[str, tuple[int, int]] = {
    "widget": (24, 10),
    "gadget": (7, 5),
    "sprocket": (3, 8),
}


def get_inventory(product: str) -> InventoryResult:
    """Return mock inventory for ``product`` (matched without case sensitivity).

    Raises:
        KeyError: If the product is not present in the mock inventory.
    """
    normalized_product = product.strip().lower()
    try:
        quantity, reorder_level = _INVENTORY[normalized_product]
    except KeyError:
        raise KeyError(f"Unknown product: {product}") from None

    return InventoryResult(
        product=normalized_product,
        quantity=quantity,
        reorder_level=reorder_level,
    )
