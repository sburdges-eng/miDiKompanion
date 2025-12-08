#!/usr/bin/env python3
"""
Test the CSV processing and Excel generation functionality
"""

import pandas as pd
import os
from datetime import datetime
from modules.vendor_analysis.csv_processor import VendorCSVProcessor

def create_test_csvs():
    """Create sample CSV files for testing"""

    # Sample SYSCO data
    sysco_data = {
        'item': ['Garlic Powder', 'Black Pepper Ground', 'Onion Powder', 'Paprika'],
        'sysco_pack': ['3/6LB', '6/1LB', '6/1LB', '6/1LB'],
        'sysco_case_price': [213.19, 295.89, 148.95, 165.99],
        'sysco_per_lb': [11.84, 49.32, 24.83, 27.67],
        'sysco_split_price': [None, None, None, None],
        'sysco_split_per_lb': [None, None, None, None]
    }

    # Sample Shamrock data
    shamrock_data = {
        'item': ['Garlic Powder', 'Black Pepper Ground', 'Onion Powder', 'Paprika'],
        'shamrock_pack': ['1/6/LB', '25 LB', '25 LB', '25 LB'],
        'shamrock_case_price': [54.26, 95.88, 39.80, 67.47],
        'shamrock_per_lb': [9.04, 3.84, 1.59, 2.70]
    }

    # Create data directory if it doesn't exist
    os.makedirs('./data/uploads', exist_ok=True)

    # Save test CSVs
    sysco_df = pd.DataFrame(sysco_data)
    shamrock_df = pd.DataFrame(shamrock_data)

    sysco_path = './data/uploads/sysco_test_data.csv'
    shamrock_path = './data/uploads/shamrock_test_data.csv'

    sysco_df.to_csv(sysco_path, index=False)
    shamrock_df.to_csv(shamrock_path, index=False)

    print(f"✅ Created test CSV files:")
    print(f"   {sysco_path}")
    print(f"   {shamrock_path}")

    return [sysco_path, shamrock_path]

def test_csv_processing():
    """Test the CSV processing functionality"""
    print("\n🧪 TESTING CSV PROCESSING FUNCTIONALITY")
    print("=" * 50)

    # Create test data
    csv_files = create_test_csvs()

    # Initialize processor
    processor = VendorCSVProcessor()

    # Test combining CSVs
    print("\n📊 Combining CSV files...")
    combined_df = processor.combine_vendor_csvs(csv_files)

    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(f"Columns: {list(combined_df.columns)}")

    if not combined_df.empty:
        print("\nFirst few rows:")
        print(combined_df.head())

        # Test Excel generation
        print("\n📊 Generating Excel report...")
        excel_path = f"./data/exports/test_vendor_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        os.makedirs('./data/exports', exist_ok=True)

        processor.generate_comparison_excel(combined_df, excel_path)
        print(f"✅ Excel report generated: {excel_path}")

        # Verify Excel file was created
        if os.path.exists(excel_path):
            print("✅ Excel file exists and is ready for download")
        else:
            print("❌ Excel file was not created")

    print("\n✅ CSV Processing Test Complete")

if __name__ == "__main__":
    test_csv_processing()