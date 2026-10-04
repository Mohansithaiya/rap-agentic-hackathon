"""Practice tools backed by deterministic mock data."""

from pydantic import BaseModel


class InventoryResult(BaseModel):
    """Inventory details for a product."""

    product: str
    quantity: int
    reorder_level: int


class ProductDetailsResult(BaseModel):
    """Basic details for a product."""

    product: str
    category: str
    description: str
    unit_price: float


_INVENTORY: dict[str, tuple[int, int]] = {
    "widget": (24, 10),
    "gadget": (7, 5),
    "sprocket": (3, 8),
}

_PRODUCT_DETAILS: dict[str, tuple[str, str, float]] = {
    "widget": ("Hardware", "Standard widget for general use.", 12.50),
    "gadget": ("Electronics", "Compact multi-purpose gadget.", 29.99),
    "sprocket": ("Hardware", "Precision sprocket for mechanical assemblies.", 8.75),
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


def get_product_details(product: str) -> ProductDetailsResult:
    """Return mock details for ``product`` (matched without case sensitivity).

    Raises:
        KeyError: If the product is not present in the mock dataset.
    """
    normalized_product = product.strip().lower()
    try:
        category, description, unit_price = _PRODUCT_DETAILS[normalized_product]
    except KeyError:
        raise KeyError(f"Unknown product: {product}") from None

    return ProductDetailsResult(
        product=normalized_product,
        category=category,
        description=description,
        unit_price=unit_price,
    )
