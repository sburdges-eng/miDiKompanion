"""
Vendor CSV Upload and Processing Service
Handles uploading, combining, and analyzing vendor CSV files
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import uuid
from werkzeug.utils import secure_filename

class VendorCSVProcessor:
    """Process and combine vendor CSV files with automatic analysis"""

    def __init__(self, upload_dir: str = "./data/uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # Expected CSV columns for different vendor formats
        self.vendor_columns = {
            'sysco': ['item', 'sysco_pack', 'sysco_case_price', 'sysco_per_lb', 'sysco_split_price', 'sysco_split_per_lb'],
            'shamrock': ['item', 'shamrock_pack', 'shamrock_case_price', 'shamrock_per_lb'],
            'us_foods': ['item', 'us_foods_pack', 'us_foods_case_price', 'us_foods_per_lb'],
            'restaurant_depot': ['item', 'restaurant_depot_pack', 'restaurant_depot_case_price', 'restaurant_depot_per_lb']
        }

    def process_uploaded_csv(self, file, vendor_name: str) -> Dict:
        """
        Process an uploaded CSV file and return analysis results

        Args:
            file: Flask file object
            vendor_name: Name of the vendor (sysco, shamrock, etc.)

        Returns:
            Dict with processing results and file info
        """
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{vendor_name}_{timestamp}_{secure_filename(file.filename)}"
        filepath = self.upload_dir / filename

        # Save file
        file.save(filepath)

        try:
            # Read and validate CSV
            df = pd.read_csv(filepath)

            # Basic validation
            if df.empty:
                return {
                    'success': False,
                    'error': 'CSV file is empty',
                    'filename': filename
                }

            # Analyze the data
            analysis = self.analyze_vendor_csv(df, vendor_name)

            return {
                'success': True,
                'filename': filename,
                'filepath': str(filepath),
                'rows': len(df),
                'columns': list(df.columns),
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Error processing CSV: {str(e)}',
                'filename': filename
            }

    def analyze_vendor_csv(self, df: pd.DataFrame, vendor_name: str) -> Dict:
        """Analyze a vendor CSV for data quality and insights"""

        analysis = {
            'total_items': len(df),
            'price_range': {},
            'pack_sizes': {},
            'data_quality': {}
        }

        # Price analysis
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        for col in price_cols:
            if col in df.columns:
                prices = pd.to_numeric(df[col], errors='coerce').dropna()
                if not prices.empty:
                    analysis['price_range'][col] = {
                        'min': float(prices.min()),
                        'max': float(prices.max()),
                        'avg': float(prices.mean()),
                        'median': float(prices.median())
                    }

        # Pack size analysis
        pack_cols = [col for col in df.columns if 'pack' in col.lower()]
        for col in pack_cols:
            if col in df.columns:
                unique_packs = df[col].value_counts().head(10).to_dict()
                analysis['pack_sizes'][col] = unique_packs

        # Data quality checks
        analysis['data_quality'] = {
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_items': df.duplicated(subset=['item']).sum() if 'item' in df.columns else 0,
            'zero_prices': sum((df[price_cols] == 0).sum()) if price_cols else 0
        }

        return analysis

    def combine_vendor_csvs(self, csv_files: List[str]) -> pd.DataFrame:
        """
        Combine multiple vendor CSV files into a unified comparison format

        Args:
            csv_files: List of CSV file paths

        Returns:
            Combined DataFrame with all vendor comparisons
        """
        vendor_data = {}

        for filepath in csv_files:
            try:
                df = pd.read_csv(filepath)

                # Extract vendor name from filename
                vendor_name = Path(filepath).name.split('_')[0].lower()

                # Rename columns to include vendor prefix (avoid double prefixing)
                rename_dict = {'item': 'item'}
                for col in df.columns:
                    if col != 'item':
                        rename_dict[col] = f"{vendor_name}_{col}"

                df_renamed = df.rename(columns=rename_dict)
                vendor_data[vendor_name] = df_renamed

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                continue

        if not vendor_data:
            return pd.DataFrame()

        # Start with the first vendor's data
        first_vendor = list(vendor_data.keys())[0]
        combined_df = vendor_data[first_vendor].copy()

        # Merge with other vendors
        for vendor_name, df in vendor_data.items():
            if vendor_name == first_vendor:
                continue

            # Merge on item column
            combined_df = combined_df.merge(df, on='item', how='outer')

        return combined_df

    def generate_comparison_excel(self, combined_df: pd.DataFrame, output_path: str) -> str:
        """
        Generate Excel file with three sheets:
        1. Cheaper options only
        2. More expensive with Shamrock
        3. More expensive with SYSCO

        Args:
            combined_df: Combined vendor comparison DataFrame
            output_path: Path to save Excel file

        Returns:
            Path to generated Excel file
        """
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

            # Sheet 1: Cheaper Options Only
            cheaper_df = self._get_cheaper_options(combined_df)
            if not cheaper_df.empty:
                cheaper_df.to_excel(writer, sheet_name='Cheaper_Options', index=False)
            else:
                # Create empty sheet with headers
                pd.DataFrame(columns=['item', 'cheaper_vendor', 'sysco_per_lb', 'shamrock_per_lb', 'price_difference', 'savings_percent', 'monthly_savings_est']).to_excel(writer, sheet_name='Cheaper_Options', index=False)

            # Sheet 2: More Expensive with Shamrock
            shamrock_expensive_df = self._get_shamrock_expensive(combined_df)
            if not shamrock_expensive_df.empty:
                shamrock_expensive_df.to_excel(writer, sheet_name='Shamrock_More_Expensive', index=False)
            else:
                pd.DataFrame(columns=['item', 'sysco_per_lb', 'shamrock_per_lb', 'shamrock_premium', 'premium_percent', 'monthly_cost_increase']).to_excel(writer, sheet_name='Shamrock_More_Expensive', index=False)

            # Sheet 3: More Expensive with SYSCO
            sysco_expensive_df = self._get_sysco_expensive(combined_df)
            if not sysco_expensive_df.empty:
                sysco_expensive_df.to_excel(writer, sheet_name='SYSCO_More_Expensive', index=False)
            else:
                pd.DataFrame(columns=['item', 'shamrock_per_lb', 'sysco_per_lb', 'sysco_premium', 'premium_percent', 'monthly_cost_increase']).to_excel(writer, sheet_name='SYSCO_More_Expensive', index=False)

            # Summary sheet
            summary_df = self._create_summary_sheet(combined_df)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        return output_path

    def _get_cheaper_options(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract items where one vendor is clearly cheaper"""
        results = []

        for _, row in df.iterrows():
            item = row.get('item', '')

            # Get prices per lb for comparison
            sysco_price = self._extract_price_per_lb(row, 'sysco')
            shamrock_price = self._extract_price_per_lb(row, 'shamrock')

            if sysco_price and shamrock_price:
                cheaper_vendor = 'SYSCO' if sysco_price < shamrock_price else 'Shamrock'
                price_diff = abs(sysco_price - shamrock_price)
                savings_pct = (price_diff / max(sysco_price, shamrock_price)) * 100

                results.append({
                    'item': item,
                    'cheaper_vendor': cheaper_vendor,
                    'sysco_per_lb': sysco_price,
                    'shamrock_per_lb': shamrock_price,
                    'price_difference': price_diff,
                    'savings_percent': savings_pct,
                    'monthly_savings_est': price_diff * 10  # Assume 10 lbs/month
                })

        if results:
            return pd.DataFrame(results).sort_values('savings_percent', ascending=False)
        else:
            return pd.DataFrame(columns=['item', 'cheaper_vendor', 'sysco_per_lb', 'shamrock_per_lb', 'price_difference', 'savings_percent', 'monthly_savings_est'])

    def _get_shamrock_expensive(self, df: pd.DataFrame) -> pd.DataFrame:
        """Items where Shamrock is more expensive than SYSCO"""
        results = []

        for _, row in df.iterrows():
            item = row.get('item', '')

            sysco_price = self._extract_price_per_lb(row, 'sysco')
            shamrock_price = self._extract_price_per_lb(row, 'shamrock')

            if sysco_price and shamrock_price and shamrock_price > sysco_price:
                price_diff = shamrock_price - sysco_price
                premium_pct = (price_diff / sysco_price) * 100

                results.append({
                    'item': item,
                    'sysco_per_lb': sysco_price,
                    'shamrock_per_lb': shamrock_price,
                    'shamrock_premium': price_diff,
                    'premium_percent': premium_pct,
                    'monthly_cost_increase': price_diff * 10
                })

        if results:
            return pd.DataFrame(results).sort_values('premium_percent', ascending=False)
        else:
            return pd.DataFrame(columns=['item', 'sysco_per_lb', 'shamrock_per_lb', 'shamrock_premium', 'premium_percent', 'monthly_cost_increase'])

    def _get_sysco_expensive(self, df: pd.DataFrame) -> pd.DataFrame:
        """Items where SYSCO is more expensive than Shamrock"""
        results = []

        for _, row in df.iterrows():
            item = row.get('item', '')

            sysco_price = self._extract_price_per_lb(row, 'sysco')
            shamrock_price = self._extract_price_per_lb(row, 'shamrock')

            if sysco_price and shamrock_price and sysco_price > shamrock_price:
                price_diff = sysco_price - shamrock_price
                premium_pct = (price_diff / shamrock_price) * 100

                results.append({
                    'item': item,
                    'shamrock_per_lb': shamrock_price,
                    'sysco_per_lb': sysco_price,
                    'sysco_premium': price_diff,
                    'premium_percent': premium_pct,
                    'monthly_cost_increase': price_diff * 10
                })

        if results:
            return pd.DataFrame(results).sort_values('premium_percent', ascending=False)
        else:
            return pd.DataFrame(columns=['item', 'shamrock_per_lb', 'sysco_per_lb', 'sysco_premium', 'premium_percent', 'monthly_cost_increase'])

    def _extract_price_per_lb(self, row: pd.Series, vendor: str) -> Optional[float]:
        """Extract price per pound for a vendor from row data"""
        # Try different column name patterns
        possible_cols = [
            f'{vendor}_per_lb',
            f'{vendor}_price_per_lb',
            f'{vendor}_case_price',  # Would need pack size conversion
        ]

        for col in possible_cols:
            if col in row.index and pd.notna(row[col]):
                try:
                    return float(row[col])
                except (ValueError, TypeError):
                    continue

        return None

    def _create_summary_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics for the Excel file"""
        summary_data = []

        # Overall statistics
        cheaper_options = self._get_cheaper_options(df)
        shamrock_expensive = self._get_shamrock_expensive(df)
        sysco_expensive = self._get_sysco_expensive(df)

        summary_data.append({
            'metric': 'Total Items Compared',
            'value': len(df),
            'notes': ''
        })

        summary_data.append({
            'metric': 'Items with Clear Winner',
            'value': len(cheaper_options),
            'notes': f"{len(cheaper_options)/len(df)*100:.1f}% of items"
        })

        if not cheaper_options.empty:
            summary_data.append({
                'metric': 'Average Savings %',
                'value': cheaper_options['savings_percent'].mean(),
                'notes': f"Range: {cheaper_options['savings_percent'].min():.1f}% - {cheaper_options['savings_percent'].max():.1f}%"
            })

            summary_data.append({
                'metric': 'Total Monthly Savings Est.',
                'value': cheaper_options['monthly_savings_est'].sum(),
                'notes': 'Based on 10 lbs/month per item'
            })

        summary_data.append({
            'metric': 'Items where Shamrock is More Expensive',
            'value': len(shamrock_expensive),
            'notes': f"{len(shamrock_expensive)/len(df)*100:.1f}% of items"
        })

        summary_data.append({
            'metric': 'Items where SYSCO is More Expensive',
            'value': len(sysco_expensive),
            'notes': f"{len(sysco_expensive)/len(df)*100:.1f}% of items"
        })

        return pd.DataFrame(summary_data)
