from app import app, db

with app.app_context():
    # Drop all tables
    db.drop_all()
    print("All tables dropped!")
    
    # Create all tables with the updated schema
    db.create_all()
    print("All tables recreated!") 