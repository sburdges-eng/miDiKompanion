#!/usr/bin/env python
"""
The Lariat Bible - Main Application
Restaurant Management System Web Interface
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import os
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
CORS(app)

# Import modules (when implemented)
try:
    from modules.vendor_analysis import VendorComparator, VendorCSVProcessor
    vendor_comparator = VendorComparator()
    csv_processor = VendorCSVProcessor()
except ImportError:
    vendor_comparator = None
    csv_processor = None

@app.route('/')
def index():
    """Main dashboard"""
    return jsonify({
        'message': 'Welcome to The Lariat Bible',
        'status': 'operational',
        'modules': {
            'vendor_analysis': 'ready' if vendor_comparator else 'pending',
            'inventory': 'pending',
            'recipes': 'pending',
            'catering': 'pending',
            'maintenance': 'pending',
            'reporting': 'pending'
        },
        'metrics': {
            'monthly_catering_revenue': 28000,
            'monthly_restaurant_revenue': 20000,
            'potential_annual_savings': 52000
        }
    })

@app.route('/api/vendor-comparison')
def vendor_comparison():
    """Get vendor comparison data"""
    if vendor_comparator:
        savings = vendor_comparator.compare_vendors('Shamrock Foods', 'SYSCO')
        margin_impact = vendor_comparator.calculate_margin_impact(savings)
        
        return jsonify({
            'monthly_savings': savings,
            'annual_savings': savings * 12,
            'margin_impact': margin_impact,
            'timestamp': datetime.now().isoformat()
        })
    
    return jsonify({'error': 'Vendor analysis module not available'}), 503

@app.route('/api/csv/upload', methods=['POST'])
def upload_csv():
    """Upload and process vendor CSV files"""
    if not csv_processor:
        return jsonify({'error': 'CSV processor not available'}), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    vendor_name = request.form.get('vendor_name', 'unknown')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'File must be a CSV'}), 400
    
    try:
        result = csv_processor.process_uploaded_csv(file, vendor_name)
        
        if result['success']:
            return jsonify({
                'message': 'CSV uploaded and processed successfully',
                'data': result
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/csv/generate-excel', methods=['POST'])
def generate_excel_report():
    """Generate Excel report from uploaded CSVs"""
    if not csv_processor:
        return jsonify({'error': 'CSV processor not available'}), 503
    
    try:
        # Get list of uploaded CSV files
        upload_dir = Path('./data/uploads')
        if not upload_dir.exists():
            return jsonify({'error': 'No uploaded files found'}), 404
        
        csv_files = list(upload_dir.glob('*.csv'))
        if not csv_files:
            return jsonify({'error': 'No CSV files found in uploads directory'}), 404
        
        # Combine the CSVs
        combined_df = csv_processor.combine_vendor_csvs([str(f) for f in csv_files])
        
        if combined_df.empty:
            return jsonify({'error': 'No data to process'}), 400
        
        # Generate Excel file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"vendor_comparison_{timestamp}.xlsx"
        excel_path = f"./data/exports/{excel_filename}"
        
        # Ensure exports directory exists
        Path('./data/exports').mkdir(parents=True, exist_ok=True)
        
        csv_processor.generate_comparison_excel(combined_df, excel_path)
        
        return jsonify({
            'message': 'Excel report generated successfully',
            'filename': excel_filename,
            'filepath': excel_path,
            'sheets': ['Cheaper_Options', 'Shamrock_More_Expensive', 'SYSCO_More_Expensive', 'Summary'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Excel generation failed: {str(e)}'}), 500

@app.route('/api/csv/list-uploads')
def list_uploaded_csvs():
    """List all uploaded CSV files"""
    upload_dir = Path('./data/uploads')
    if not upload_dir.exists():
        return jsonify({'files': []})
    
    files = []
    for csv_file in upload_dir.glob('*.csv'):
        stat = csv_file.stat()
        files.append({
            'filename': csv_file.name,
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'vendor': csv_file.name.split('_')[0] if '_' in csv_file.name else 'unknown'
        })
    
    return jsonify({
        'files': sorted(files, key=lambda x: x['modified'], reverse=True),
        'total': len(files)
    })

@app.route('/api/csv/delete/<filename>', methods=['DELETE'])
def delete_csv(filename):
    """Delete an uploaded CSV file"""
    filepath = Path('./data/uploads') / filename
    if filepath.exists():
        filepath.unlink()
        return jsonify({'message': f'File {filename} deleted successfully'})
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'restaurant': os.getenv('RESTAURANT_NAME', 'The Lariat')
    })

@app.route('/api/modules')
def list_modules():
    """List all available modules and their status"""
    modules = [
        {
            'name': 'Vendor Analysis',
            'endpoint': '/vendor-analysis',
            'status': 'active' if vendor_comparator else 'development',
            'description': 'Compare vendor prices and identify savings'
        },
        {
            'name': 'Inventory Management',
            'endpoint': '/inventory',
            'status': 'development',
            'description': 'Track stock levels and automate ordering'
        },
        {
            'name': 'Recipe Management',
            'endpoint': '/recipes',
            'status': 'development',
            'description': 'Standardize recipes and calculate costs'
        },
        {
            'name': 'Catering Operations',
            'endpoint': '/catering',
            'status': 'development',
            'description': 'Manage catering quotes and events'
        },
        {
            'name': 'Maintenance Tracking',
            'endpoint': '/maintenance',
            'status': 'development',
            'description': 'Schedule and track equipment maintenance'
        },
        {
            'name': 'Reporting Dashboard',
            'endpoint': '/reports',
            'status': 'development',
            'description': 'Business intelligence and analytics'
        }
    ]
    
    return jsonify(modules)

if __name__ == '__main__':
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"\n🤠 The Lariat Bible - Starting server...")
    print(f"📍 Access at: http://{host}:{port}")
    print(f"📊 API Health: http://{host}:{port}/api/health")
    print(f"🔧 Debug mode: {debug}")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(host=host, port=port, debug=debug)
