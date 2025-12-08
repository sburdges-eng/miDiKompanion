"""
Invoice Processor
Handles invoice OCR, parsing, and data extraction for vendor analysis
"""

import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class InvoiceLineItem:
    """Represents a single line item from an invoice"""
    product_code: str
    description: str
    pack_size: str
    quantity: int
    unit_price: float
    extended_price: float
    category: Optional[str] = None
    
    @property
    def price_per_unit(self) -> float:
        """Calculate price per unit ordered"""
        return self.extended_price / self.quantity if self.quantity > 0 else 0.0


@dataclass
class Invoice:
    """Represents a parsed vendor invoice"""
    invoice_number: str
    vendor_name: str
    invoice_date: datetime
    due_date: Optional[datetime]
    subtotal: float
    tax: float
    total: float
    line_items: List[InvoiceLineItem] = field(default_factory=list)
    raw_text: Optional[str] = None
    file_path: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert invoice to dictionary for JSON serialization"""
        return {
            'invoice_number': self.invoice_number,
            'vendor_name': self.vendor_name,
            'invoice_date': self.invoice_date.isoformat() if self.invoice_date else None,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'subtotal': self.subtotal,
            'tax': self.tax,
            'total': self.total,
            'line_items': [
                {
                    'product_code': item.product_code,
                    'description': item.description,
                    'pack_size': item.pack_size,
                    'quantity': item.quantity,
                    'unit_price': item.unit_price,
                    'extended_price': item.extended_price,
                    'category': item.category
                }
                for item in self.line_items
            ],
            'file_path': self.file_path
        }


class InvoiceProcessor:
    """
    Process vendor invoices for price comparison and analysis
    
    Supports:
    - Manual data entry
    - CSV/Excel imports
    - Future: OCR from scanned invoices
    """
    
    # Vendor-specific patterns for parsing
    VENDOR_PATTERNS = {
        'SYSCO': {
            'invoice_number': r'Invoice\s*#?\s*(\d+)',
            'date': r'Invoice\s+Date[:\s]*(\d{1,2}/\d{1,2}/\d{2,4})',
            'line_item': r'(\d{6,7})\s+(.+?)\s+(\d+/\d+\s*(?:LB|OZ|GAL|EA))\s+(\d+)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)'
        },
        'Shamrock Foods': {
            'invoice_number': r'Invoice[:\s]*(\d+)',
            'date': r'Date[:\s]*(\d{1,2}/\d{1,2}/\d{2,4})',
            'line_item': r'(\d{5,7})\s+(.+?)\s+(\d+/\d+/(?:LB|OZ|GAL))\s+(\d+)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)'
        }
    }
    
    def __init__(self, storage_path: str = None):
        """
        Initialize the invoice processor
        
        Args:
            storage_path: Path to store processed invoice data
        """
        self.storage_path = storage_path or './data/invoices'
        self.processed_invoices: List[Invoice] = []
        self.price_history: Dict[str, List[Dict]] = {}  # product_code -> list of prices
        
        # Ensure storage directory exists
        Path(self.storage_path).mkdir(parents=True, exist_ok=True)
    
    def parse_invoice_text(self, text: str, vendor: str = None) -> Optional[Invoice]:
        """
        Parse invoice data from text (manual entry or OCR output)
        
        Args:
            text: Raw invoice text
            vendor: Vendor name (auto-detected if not provided)
            
        Returns:
            Parsed Invoice object or None if parsing fails
        """
        # Auto-detect vendor if not provided
        if not vendor:
            vendor = self._detect_vendor(text)
        
        if not vendor or vendor not in self.VENDOR_PATTERNS:
            print(f"Warning: Unknown vendor format. Using generic parsing.")
            return self._generic_parse(text)
        
        patterns = self.VENDOR_PATTERNS[vendor]
        
        # Extract invoice number
        invoice_match = re.search(patterns['invoice_number'], text, re.IGNORECASE)
        invoice_number = invoice_match.group(1) if invoice_match else f"INV-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Extract date
        date_match = re.search(patterns['date'], text, re.IGNORECASE)
        invoice_date = self._parse_date(date_match.group(1)) if date_match else datetime.now()
        
        # Extract line items
        line_items = []
        for match in re.finditer(patterns['line_item'], text):
            item = InvoiceLineItem(
                product_code=match.group(1),
                description=match.group(2).strip(),
                pack_size=match.group(3),
                quantity=int(match.group(4)),
                unit_price=float(match.group(5).replace(',', '')),
                extended_price=float(match.group(6).replace(',', ''))
            )
            line_items.append(item)
        
        # Calculate totals
        subtotal = sum(item.extended_price for item in line_items)
        
        return Invoice(
            invoice_number=invoice_number,
            vendor_name=vendor,
            invoice_date=invoice_date,
            due_date=None,
            subtotal=subtotal,
            tax=0.0,  # Tax detection would need additional pattern
            total=subtotal,
            line_items=line_items,
            raw_text=text
        )
    
    def import_from_csv(self, file_path: str, vendor: str) -> Invoice:
        """
        Import invoice data from CSV file
        
        Expected columns:
        - product_code, description, pack_size, quantity, unit_price, extended_price
        
        Args:
            file_path: Path to CSV file
            vendor: Vendor name
            
        Returns:
            Parsed Invoice object
        """
        import csv
        
        line_items = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = InvoiceLineItem(
                    product_code=row.get('product_code', row.get('code', '')),
                    description=row.get('description', row.get('item', '')),
                    pack_size=row.get('pack_size', row.get('pack', '')),
                    quantity=int(row.get('quantity', row.get('qty', 1))),
                    unit_price=float(row.get('unit_price', row.get('price', 0))),
                    extended_price=float(row.get('extended_price', row.get('total', 0))),
                    category=row.get('category', None)
                )
                line_items.append(item)
        
        subtotal = sum(item.extended_price for item in line_items)
        
        invoice = Invoice(
            invoice_number=f"CSV-{Path(file_path).stem}",
            vendor_name=vendor,
            invoice_date=datetime.now(),
            due_date=None,
            subtotal=subtotal,
            tax=0.0,
            total=subtotal,
            line_items=line_items,
            file_path=file_path
        )
        
        self.processed_invoices.append(invoice)
        self._update_price_history(invoice)
        
        return invoice
    
    def add_manual_entry(self, vendor: str, items: List[Dict], 
                         invoice_date: datetime = None) -> Invoice:
        """
        Add invoice data via manual entry
        
        Args:
            vendor: Vendor name
            items: List of item dictionaries with keys:
                   product_code, description, pack_size, quantity, unit_price
            invoice_date: Date of invoice (defaults to now)
            
        Returns:
            Created Invoice object
        """
        line_items = []
        
        for item in items:
            quantity = item.get('quantity', 1)
            unit_price = item.get('unit_price', 0)
            
            line_item = InvoiceLineItem(
                product_code=item.get('product_code', ''),
                description=item.get('description', ''),
                pack_size=item.get('pack_size', ''),
                quantity=quantity,
                unit_price=unit_price,
                extended_price=item.get('extended_price', quantity * unit_price),
                category=item.get('category', None)
            )
            line_items.append(line_item)
        
        subtotal = sum(item.extended_price for item in line_items)
        
        invoice = Invoice(
            invoice_number=f"MANUAL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            vendor_name=vendor,
            invoice_date=invoice_date or datetime.now(),
            due_date=None,
            subtotal=subtotal,
            tax=0.0,
            total=subtotal,
            line_items=line_items
        )
        
        self.processed_invoices.append(invoice)
        self._update_price_history(invoice)
        
        return invoice
    
    def get_price_history(self, product_code: str) -> List[Dict]:
        """
        Get price history for a specific product
        
        Args:
            product_code: Product code to look up
            
        Returns:
            List of historical prices with dates and vendors
        """
        return self.price_history.get(product_code, [])
    
    def compare_invoices(self, invoice1: Invoice, invoice2: Invoice) -> Dict:
        """
        Compare two invoices from different vendors
        
        Args:
            invoice1: First invoice
            invoice2: Second invoice
            
        Returns:
            Comparison results with matching products and price differences
        """
        # Create lookup by description (normalized)
        def normalize(desc: str) -> str:
            return re.sub(r'[^a-z0-9]', '', desc.lower())
        
        items1 = {normalize(item.description): item for item in invoice1.line_items}
        items2 = {normalize(item.description): item for item in invoice2.line_items}
        
        matches = []
        only_in_1 = []
        only_in_2 = []
        
        for key, item1 in items1.items():
            if key in items2:
                item2 = items2[key]
                matches.append({
                    'description': item1.description,
                    f'{invoice1.vendor_name}_price': item1.unit_price,
                    f'{invoice2.vendor_name}_price': item2.unit_price,
                    'difference': item1.unit_price - item2.unit_price,
                    'cheaper_vendor': invoice1.vendor_name if item1.unit_price < item2.unit_price else invoice2.vendor_name
                })
            else:
                only_in_1.append(item1.description)
        
        for key in items2:
            if key not in items1:
                only_in_2.append(items2[key].description)
        
        total_savings = sum(abs(m['difference']) for m in matches if m['difference'] != 0)
        
        return {
            'matched_products': len(matches),
            'matches': matches,
            f'only_in_{invoice1.vendor_name}': only_in_1,
            f'only_in_{invoice2.vendor_name}': only_in_2,
            'potential_savings': total_savings
        }
    
    def save_invoice(self, invoice: Invoice, filename: str = None) -> str:
        """
        Save processed invoice to storage
        
        Args:
            invoice: Invoice to save
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if not filename:
            filename = f"{invoice.vendor_name}_{invoice.invoice_number}_{invoice.invoice_date.strftime('%Y%m%d')}.json"
        
        # Sanitize filename
        filename = re.sub(r'[^a-zA-Z0-9_\-.]', '_', filename)
        file_path = os.path.join(self.storage_path, filename)
        
        with open(file_path, 'w') as f:
            json.dump(invoice.to_dict(), f, indent=2)
        
        return file_path
    
    def load_invoice(self, file_path: str) -> Invoice:
        """
        Load invoice from storage
        
        Args:
            file_path: Path to invoice JSON file
            
        Returns:
            Loaded Invoice object
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        line_items = [
            InvoiceLineItem(**item) for item in data.get('line_items', [])
        ]
        
        return Invoice(
            invoice_number=data['invoice_number'],
            vendor_name=data['vendor_name'],
            invoice_date=datetime.fromisoformat(data['invoice_date']) if data['invoice_date'] else None,
            due_date=datetime.fromisoformat(data['due_date']) if data.get('due_date') else None,
            subtotal=data['subtotal'],
            tax=data['tax'],
            total=data['total'],
            line_items=line_items,
            file_path=file_path
        )
    
    def generate_price_report(self, vendor: str = None) -> Dict:
        """
        Generate a price report from all processed invoices
        
        Args:
            vendor: Filter by vendor (optional)
            
        Returns:
            Report with price statistics
        """
        invoices = self.processed_invoices
        if vendor:
            invoices = [inv for inv in invoices if inv.vendor_name == vendor]
        
        if not invoices:
            return {'error': 'No invoices found'}
        
        all_items = []
        for invoice in invoices:
            for item in invoice.line_items:
                all_items.append({
                    'vendor': invoice.vendor_name,
                    'date': invoice.invoice_date,
                    'product_code': item.product_code,
                    'description': item.description,
                    'unit_price': item.unit_price
                })
        
        # Group by product and calculate statistics
        products = {}
        for item in all_items:
            key = item['product_code'] or item['description']
            if key not in products:
                products[key] = {
                    'description': item['description'],
                    'prices': [],
                    'vendors': set()
                }
            products[key]['prices'].append(item['unit_price'])
            products[key]['vendors'].add(item['vendor'])
        
        report = {
            'total_invoices': len(invoices),
            'total_products': len(products),
            'products': {}
        }
        
        for key, data in products.items():
            prices = data['prices']
            report['products'][key] = {
                'description': data['description'],
                'vendors': list(data['vendors']),
                'min_price': min(prices),
                'max_price': max(prices),
                'avg_price': sum(prices) / len(prices),
                'price_variance': max(prices) - min(prices)
            }
        
        return report
    
    def _detect_vendor(self, text: str) -> Optional[str]:
        """Auto-detect vendor from invoice text"""
        text_lower = text.lower()
        
        if 'sysco' in text_lower:
            return 'SYSCO'
        elif 'shamrock' in text_lower:
            return 'Shamrock Foods'
        elif 'us foods' in text_lower:
            return 'US Foods'
        elif 'performance food' in text_lower:
            return 'Performance Food Group'
        
        return None
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime"""
        formats = [
            '%m/%d/%Y',
            '%m/%d/%y',
            '%Y-%m-%d',
            '%d-%m-%Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return datetime.now()
    
    def _generic_parse(self, text: str) -> Optional[Invoice]:
        """Generic parsing for unknown vendor formats"""
        # Very basic parsing - just extract what we can
        return Invoice(
            invoice_number=f"GENERIC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            vendor_name="Unknown",
            invoice_date=datetime.now(),
            due_date=None,
            subtotal=0.0,
            tax=0.0,
            total=0.0,
            line_items=[],
            raw_text=text
        )
    
    def _update_price_history(self, invoice: Invoice):
        """Update price history with invoice data"""
        for item in invoice.line_items:
            if item.product_code not in self.price_history:
                self.price_history[item.product_code] = []
            
            self.price_history[item.product_code].append({
                'date': invoice.invoice_date.isoformat(),
                'vendor': invoice.vendor_name,
                'price': item.unit_price,
                'pack_size': item.pack_size
            })


# Example usage and testing
if __name__ == "__main__":
    processor = InvoiceProcessor()
    
    # Example manual entry
    sample_items = [
        {
            'product_code': '1234567',
            'description': 'Black Pepper Ground',
            'pack_size': '6/1LB',
            'quantity': 2,
            'unit_price': 49.29
        },
        {
            'product_code': '2345678',
            'description': 'Garlic Powder',
            'pack_size': '3/6LB',
            'quantity': 1,
            'unit_price': 213.19
        }
    ]
    
    invoice = processor.add_manual_entry('SYSCO', sample_items)
    print(f"Created invoice: {invoice.invoice_number}")
    print(f"Subtotal: ${invoice.subtotal:.2f}")
    print(f"Items: {len(invoice.line_items)}")
    
    # Save the invoice
    saved_path = processor.save_invoice(invoice)
    print(f"Saved to: {saved_path}")
