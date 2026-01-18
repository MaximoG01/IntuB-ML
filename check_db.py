import sys
sys.path.insert(0, '.')

from app import create_app
from app.models import IntubationRecord

app = create_app()

with app.app_context():
    # Check actual database
    total_records = IntubationRecord.query.count()
    labeled_records = IntubationRecord.query.filter(IntubationRecord.difficult_binary.isnot(None)).count()
    
    print(f"Total records in DB: {total_records}")
    print(f"Labeled records (with difficult_binary): {labeled_records}")
    
    if labeled_records > 0:
        print("\nFirst 5 labeled records:")
        for rec in IntubationRecord.query.filter(IntubationRecord.difficult_binary.isnot(None)).limit(5).all():
            print(f"  ID={rec.id}, difficult={rec.difficult_binary}, operator={rec.operator_id}")
