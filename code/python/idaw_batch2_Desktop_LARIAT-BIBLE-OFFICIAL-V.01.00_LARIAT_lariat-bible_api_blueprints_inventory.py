from flask import Blueprint, jsonify, request
from modules.core.inventory_manager import InventoryManager

bp = Blueprint("inventory", __name__, url_prefix="/api/inventory")
manager = InventoryManager()


@bp.route("/", methods=["GET"])
def list_items():
    items = manager.list_items()
    return jsonify([{
        "uuid": item.uuid,
        "name": item.name,
        "unit": item.unit,
        "quantity_on_hand": item.quantity_on_hand,
        "unit_cost": item.unit_cost,
        "preferred_vendor": item.preferred_vendor
    } for item in items])


@bp.route("/", methods=["POST"])
def create_item():
    data = request.get_json() or {}
    item = manager.create_item(
        name=data.get("name"),
        unit=data.get("unit", "EA"),
        quantity=float(data.get("quantity", 0)),
        unit_cost=float(data.get("unit_cost", 0)),
        preferred_vendor=data.get("preferred_vendor")
    )
    return jsonify({"uuid": item.uuid, "name": item.name}), 201


@bp.route("/<uuid_str>/adjust", methods=["POST"])
def adjust(uuid_str):
    data = request.get_json() or {}
    delta = float(data.get("delta", 0))
    mv_type = data.get("movement_type", "ADJUSTMENT")
    reference = data.get("reference")
    item = manager.adjust_stock(uuid_str, delta, movement_type=mv_type, reference=reference)
    return jsonify({"uuid": item.uuid, "quantity_on_hand": item.quantity_on_hand})
