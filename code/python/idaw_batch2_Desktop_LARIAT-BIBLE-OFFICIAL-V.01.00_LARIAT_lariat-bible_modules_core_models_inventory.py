from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from modules.core.db import Base


class InventoryItem(Base):
    __tablename__ = "inventory_items"

    id = Column(Integer, primary_key=True)
    uuid = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    unit = Column(String, default="EA")
    quantity_on_hand = Column(Float, default=0.0)
    unit_cost = Column(Float, default=0.0)
    preferred_vendor = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    movements = relationship("StockMovement", back_populates="inventory_item")


class StockMovement(Base):
    __tablename__ = "stock_movements"

    id = Column(Integer, primary_key=True)
    inventory_id = Column(Integer, ForeignKey("inventory_items.id"), nullable=False)
    quantity_delta = Column(Float, nullable=False)
    movement_type = Column(String, nullable=False)  # RECEIVE, USAGE, ADJUSTMENT, TRANSFER
    reference = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    inventory_item = relationship("InventoryItem", back_populates="movements")
