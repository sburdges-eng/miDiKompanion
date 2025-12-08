"""
Vendor Data Models
Core data structures for vendor management and price tracking
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import json


class ProductCategory(Enum):
    """Product categories for organization and reporting"""
    SPICES = "spices"
    PROTEINS = "proteins"
    PRODUCE = "produce"
    DAIRY = "dairy"
    DRY_GOODS = "dry_goods"
    BEVERAGES = "beverages"
    CHEMICALS = "chemicals"
    PAPER_GOODS = "paper_goods"
    EQUIPMENT = "equipment"
    OTHER = "other"


class VendorTier(Enum):
    """Vendor priority tiers"""
    PRIMARY = "primary"      # First choice vendor
    SECONDARY = "secondary"  # Backup vendor
    SPECIALTY = "specialty"  # For specific items only
    EMERGENCY = "emergency"  # Last resort


@dataclass
class Vendor:
    """
    Represents a food service vendor
    """
    name: str
    vendor_code: str
    tier: VendorTier = VendorTier.SECONDARY
    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None
    contact_email: Optional[str] = None
    account_number: Optional[str] = None
    payment_terms: str = "Net 30"
    delivery_days: List[str] = field(default_factory=list)
    minimum_order: float = 0.0
    delivery_fee: float = 0.0
    notes: str = ""
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'vendor_code': self.vendor_code,
            'tier': self.tier.value,
            'contact_name': self.contact_name,
            'contact_phone': self.contact_phone,
            'contact_email': self.contact_email,
            'account_number': self.account_number,
            'payment_terms': self.payment_terms,
            'delivery_days': self.delivery_days,
            'minimum_order': self.minimum_order,
            'delivery_fee': self.delivery_fee,
            'notes': self.notes,
            'active': self.active,
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Vendor':
        data['tier'] = VendorTier(data.get('tier', 'secondary'))
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class Product:
    """
    Represents a product that can be ordered from vendors
    """
    product_id: str
    name: str
    description: str
    category: ProductCategory
    
    # Standard unit for comparison (e.g., "LB", "OZ", "GAL", "EACH")
    standard_unit: str
    
    # Storage and handling
    requires_refrigeration: bool = False
    shelf_life_days: Optional[int] = None
    
    # Usage tracking
    par_level: float = 0.0  # Minimum quantity to keep on hand
    average_weekly_usage: float = 0.0
    
    # Specifications (critical for matching products between vendors)
    specifications: Dict[str, str] = field(default_factory=dict)
    
    # Alternative products
    substitutes: List[str] = field(default_factory=list)  # List of product_ids
    
    notes: str = ""
    active: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'product_id': self.product_id,
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'standard_unit': self.standard_unit,
            'requires_refrigeration': self.requires_refrigeration,
            'shelf_life_days': self.shelf_life_days,
            'par_level': self.par_level,
            'average_weekly_usage': self.average_weekly_usage,
            'specifications': self.specifications,
            'substitutes': self.substitutes,
            'notes': self.notes,
            'active': self.active
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Product':
        data['category'] = ProductCategory(data.get('category', 'other'))
        return cls(**data)


@dataclass
class VendorProduct:
    """
    Represents a specific product offering from a vendor
    Links a Product to a Vendor with vendor-specific details
    """
    vendor_code: str  # Vendor's code for this product
    product_id: str   # Reference to our Product
    vendor_id: str    # Reference to the Vendor
    
    # Vendor-specific details
    vendor_description: str
    pack_size: str
    case_price: float
    split_available: bool = False
    split_price: Optional[float] = None
    
    # Calculated fields
    price_per_unit: float = 0.0  # Calculated price per standard unit
    units_per_case: float = 0.0  # Number of standard units in a case
    
    # Ordering constraints
    minimum_order_qty: int = 1
    order_multiple: int = 1  # Must order in multiples of this
    
    # Status
    in_stock: bool = True
    lead_time_days: int = 1
    last_ordered: Optional[datetime] = None
    last_price_update: datetime = field(default_factory=datetime.now)
    
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'vendor_code': self.vendor_code,
            'product_id': self.product_id,
            'vendor_id': self.vendor_id,
            'vendor_description': self.vendor_description,
            'pack_size': self.pack_size,
            'case_price': self.case_price,
            'split_available': self.split_available,
            'split_price': self.split_price,
            'price_per_unit': self.price_per_unit,
            'units_per_case': self.units_per_case,
            'minimum_order_qty': self.minimum_order_qty,
            'order_multiple': self.order_multiple,
            'in_stock': self.in_stock,
            'lead_time_days': self.lead_time_days,
            'last_ordered': self.last_ordered.isoformat() if self.last_ordered else None,
            'last_price_update': self.last_price_update.isoformat(),
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VendorProduct':
        if 'last_ordered' in data and data['last_ordered']:
            data['last_ordered'] = datetime.fromisoformat(data['last_ordered'])
        if 'last_price_update' in data:
            data['last_price_update'] = datetime.fromisoformat(data['last_price_update'])
        return cls(**data)


@dataclass
class PriceHistory:
    """
    Tracks historical prices for a vendor product
    """
    vendor_product_id: str  # Combination of vendor_id + vendor_code
    timestamp: datetime
    case_price: float
    split_price: Optional[float] = None
    price_per_unit: float = 0.0
    source: str = "manual"  # "manual", "invoice", "order_guide", "api"
    
    def to_dict(self) -> Dict:
        return {
            'vendor_product_id': self.vendor_product_id,
            'timestamp': self.timestamp.isoformat(),
            'case_price': self.case_price,
            'split_price': self.split_price,
            'price_per_unit': self.price_per_unit,
            'source': self.source
        }


@dataclass
class PriceComparison:
    """
    Result of comparing prices between vendors for a product
    """
    product_id: str
    product_name: str
    comparison_date: datetime = field(default_factory=datetime.now)
    
    # Vendor options (list of VendorProduct entries)
    vendor_options: List[Dict] = field(default_factory=list)
    
    # Best option
    best_vendor: Optional[str] = None
    best_price_per_unit: Optional[float] = None
    
    # Savings analysis
    savings_vs_highest: float = 0.0
    savings_percent: float = 0.0
    
    # Recommendation
    recommendation: str = ""
    notes: str = ""
    
    def add_vendor_option(self, vendor_name: str, pack_size: str, 
                          case_price: float, price_per_unit: float,
                          split_available: bool = False, 
                          split_price: Optional[float] = None):
        """Add a vendor option to the comparison"""
        option = {
            'vendor': vendor_name,
            'pack_size': pack_size,
            'case_price': case_price,
            'price_per_unit': price_per_unit,
            'split_available': split_available,
            'split_price': split_price
        }
        self.vendor_options.append(option)
        self._recalculate()
    
    def _recalculate(self):
        """Recalculate best option and savings"""
        if not self.vendor_options:
            return
        
        # Find best price per unit
        sorted_options = sorted(self.vendor_options, key=lambda x: x['price_per_unit'])
        best = sorted_options[0]
        worst = sorted_options[-1]
        
        self.best_vendor = best['vendor']
        self.best_price_per_unit = best['price_per_unit']
        
        if worst['price_per_unit'] > 0:
            self.savings_vs_highest = worst['price_per_unit'] - best['price_per_unit']
            self.savings_percent = (self.savings_vs_highest / worst['price_per_unit']) * 100
        
        # Generate recommendation
        if len(self.vendor_options) == 1:
            self.recommendation = f"Only available from {best['vendor']}"
        elif self.savings_percent >= 10:
            self.recommendation = f"Strong savings ({self.savings_percent:.1f}%) with {best['vendor']}"
        elif self.savings_percent >= 5:
            self.recommendation = f"Moderate savings ({self.savings_percent:.1f}%) with {best['vendor']}"
        else:
            self.recommendation = f"Prices similar - consider delivery convenience"
    
    def to_dict(self) -> Dict:
        return {
            'product_id': self.product_id,
            'product_name': self.product_name,
            'comparison_date': self.comparison_date.isoformat(),
            'vendor_options': self.vendor_options,
            'best_vendor': self.best_vendor,
            'best_price_per_unit': self.best_price_per_unit,
            'savings_vs_highest': self.savings_vs_highest,
            'savings_percent': self.savings_percent,
            'recommendation': self.recommendation,
            'notes': self.notes
        }


class VendorDatabase:
    """
    In-memory database for vendor data
    Can be extended to use SQLite, PostgreSQL, etc.
    """
    
    def __init__(self):
        self.vendors: Dict[str, Vendor] = {}
        self.products: Dict[str, Product] = {}
        self.vendor_products: Dict[str, VendorProduct] = {}
        self.price_history: List[PriceHistory] = []
        
        # Initialize with known vendors
        self._initialize_vendors()
    
    def _initialize_vendors(self):
        """Initialize with The Lariat's known vendors"""
        vendors = [
            Vendor(
                name="Shamrock Foods",
                vendor_code="SHAMROCK",
                tier=VendorTier.PRIMARY,
                payment_terms="Net 14",
                delivery_days=["Tuesday", "Friday"],
                notes="29.5% better pricing than SYSCO on average"
            ),
            Vendor(
                name="SYSCO",
                vendor_code="SYSCO",
                tier=VendorTier.SECONDARY,
                payment_terms="Net 30",
                delivery_days=["Monday", "Wednesday", "Friday"],
                notes="Larger selection, higher prices"
            ),
            Vendor(
                name="US Foods",
                vendor_code="USFOODS",
                tier=VendorTier.SPECIALTY,
                payment_terms="Net 30",
                delivery_days=["Tuesday", "Thursday"],
                notes="Consider for specialty items"
            ),
            Vendor(
                name="Restaurant Depot",
                vendor_code="RESTDEPOT",
                tier=VendorTier.EMERGENCY,
                payment_terms="Cash",
                delivery_days=[],  # Pick-up only
                notes="Cash & carry - emergency purchases only"
            )
        ]
        
        for vendor in vendors:
            self.vendors[vendor.vendor_code] = vendor
    
    def add_vendor(self, vendor: Vendor):
        """Add a vendor to the database"""
        self.vendors[vendor.vendor_code] = vendor
    
    def get_vendor(self, vendor_code: str) -> Optional[Vendor]:
        """Get a vendor by code"""
        return self.vendors.get(vendor_code)
    
    def add_product(self, product: Product):
        """Add a product to the database"""
        self.products[product.product_id] = product
    
    def get_product(self, product_id: str) -> Optional[Product]:
        """Get a product by ID"""
        return self.products.get(product_id)
    
    def add_vendor_product(self, vendor_product: VendorProduct):
        """Add a vendor-specific product listing"""
        key = f"{vendor_product.vendor_id}:{vendor_product.vendor_code}"
        self.vendor_products[key] = vendor_product
    
    def get_vendor_products_for_product(self, product_id: str) -> List[VendorProduct]:
        """Get all vendor offerings for a product"""
        return [vp for vp in self.vendor_products.values() 
                if vp.product_id == product_id]
    
    def add_price_history(self, history: PriceHistory):
        """Add a price history entry"""
        self.price_history.append(history)
    
    def get_price_history(self, vendor_product_id: str, 
                          days: int = 90) -> List[PriceHistory]:
        """Get price history for a vendor product"""
        cutoff = datetime.now().timestamp() - (days * 86400)
        return [h for h in self.price_history 
                if h.vendor_product_id == vendor_product_id 
                and h.timestamp.timestamp() > cutoff]
    
    def compare_product_prices(self, product_id: str) -> PriceComparison:
        """Generate a price comparison for a product across all vendors"""
        product = self.get_product(product_id)
        if not product:
            return PriceComparison(
                product_id=product_id,
                product_name="Unknown",
                notes="Product not found"
            )
        
        comparison = PriceComparison(
            product_id=product_id,
            product_name=product.name
        )
        
        vendor_products = self.get_vendor_products_for_product(product_id)
        for vp in vendor_products:
            vendor = self.get_vendor(vp.vendor_id)
            if vendor and vendor.active:
                comparison.add_vendor_option(
                    vendor_name=vendor.name,
                    pack_size=vp.pack_size,
                    case_price=vp.case_price,
                    price_per_unit=vp.price_per_unit,
                    split_available=vp.split_available,
                    split_price=vp.split_price
                )
        
        return comparison
    
    def save_to_file(self, file_path: str):
        """Save database to JSON file"""
        data = {
            'vendors': {k: v.to_dict() for k, v in self.vendors.items()},
            'products': {k: v.to_dict() for k, v in self.products.items()},
            'vendor_products': {k: v.to_dict() for k, v in self.vendor_products.items()},
            'price_history': [h.to_dict() for h in self.price_history]
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, file_path: str):
        """Load database from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.vendors = {k: Vendor.from_dict(v) for k, v in data.get('vendors', {}).items()}
        self.products = {k: Product.from_dict(v) for k, v in data.get('products', {}).items()}
        self.vendor_products = {k: VendorProduct.from_dict(v) 
                                for k, v in data.get('vendor_products', {}).items()}
        # Price history would need special handling for datetime


# Example usage
if __name__ == "__main__":
    db = VendorDatabase()
    
    print("Initialized Vendors:")
    for code, vendor in db.vendors.items():
        print(f"  {code}: {vendor.name} ({vendor.tier.value})")
    
    # Add a sample product
    black_pepper = Product(
        product_id="SPICE-BP-001",
        name="Black Pepper Ground",
        description="Ground black pepper, restaurant grind",
        category=ProductCategory.SPICES,
        standard_unit="LB",
        specifications={
            'grind': 'restaurant',
            'origin': 'Vietnam/Indonesia blend'
        },
        par_level=10.0,
        average_weekly_usage=2.5
    )
    
    db.add_product(black_pepper)
    
    # Add vendor offerings
    sysco_bp = VendorProduct(
        vendor_code="1234567",
        product_id="SPICE-BP-001",
        vendor_id="SYSCO",
        vendor_description="BLACK PEPPER GROUND RESTAURANT",
        pack_size="6/1LB",
        case_price=295.89,
        split_available=True,
        split_price=52.99,
        price_per_unit=49.32,  # $295.89 / 6 lbs
        units_per_case=6.0
    )
    
    shamrock_bp = VendorProduct(
        vendor_code="78901",
        product_id="SPICE-BP-001",
        vendor_id="SHAMROCK",
        vendor_description="PEPPER BLACK GROUND REST",
        pack_size="25 LB",
        case_price=95.88,
        split_available=False,
        split_price=None,
        price_per_unit=3.84,  # $95.88 / 25 lbs
        units_per_case=25.0
    )
    
    db.add_vendor_product(sysco_bp)
    db.add_vendor_product(shamrock_bp)
    
    # Compare prices
    comparison = db.compare_product_prices("SPICE-BP-001")
    print(f"\nPrice Comparison for {comparison.product_name}:")
    print(f"  Best Vendor: {comparison.best_vendor}")
    print(f"  Best Price/LB: ${comparison.best_price_per_unit:.2f}")
    print(f"  Savings: {comparison.savings_percent:.1f}%")
    print(f"  Recommendation: {comparison.recommendation}")
