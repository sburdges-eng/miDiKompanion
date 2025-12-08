import uuid
from typing import Optional, List
from modules.core.db import get_session, engine, Base
from modules.core.models_inventory import InventoryItem, StockMovement


class InventoryManager:
    """Basic inventory manager that persists items and stock movements."""

    def __init__(self):
        # Ensure tables exist
        Base.metadata.create_all(bind=engine)

    def create_item(self, name: str, unit: str = "EA", quantity: float = 0.0, unit_cost: float = 0.0,
                    preferred_vendor: Optional[str] = None, uuid_str: Optional[str] = None) -> InventoryItem:
        if uuid_str is None:
            uuid_str = str(uuid.uuid4())
        with get_session() as session:
            item = InventoryItem(uuid=uuid_str, name=name, unit=unit, quantity_on_hand=quantity,
                                 unit_cost=unit_cost, preferred_vendor=preferred_vendor)
            session.add(item)
            session.flush()
            # initial movement when quantity provided
            if quantity != 0:
                mv = StockMovement(inventory_id=item.id, quantity_delta=quantity, movement_type="RECEIVE",
                                   reference="initial")
                session.add(mv)
            session.refresh(item)
            return item

    def get_item(self, uuid_str: str) -> Optional[InventoryItem]:
        with get_session() as session:
            return session.query(InventoryItem).filter_by(uuid=uuid_str).one_or_none()

    def list_items(self) -> List[InventoryItem]:
        with get_session() as session:
            return session.query(InventoryItem).order_by(InventoryItem.name).all()

    def adjust_stock(self, uuid_str: str, delta: float, movement_type: str = "ADJUSTMENT",
                     reference: Optional[str] = None) -> InventoryItem:
        with get_session() as session:
            item = session.query(InventoryItem).filter_by(uuid=uuid_str).with_for_update().one()
            item.quantity_on_hand = (item.quantity_on_hand or 0.0) + delta
            mv = StockMovement(inventory_id=item.id, quantity_delta=delta, movement_type=movement_type,
                               reference=reference)
            session.add(mv)
            session.flush()
            session.refresh(item)
            return item
