"""Initialize the local database for the Lariat Kitchen Data Core."""

from modules.core.db import Base, engine
from modules.core.models_inventory import InventoryItem, StockMovement


def init():
    Base.metadata.create_all(bind=engine)
    print("Initialized DB tables using:", engine)


if __name__ == "__main__":
    init()
