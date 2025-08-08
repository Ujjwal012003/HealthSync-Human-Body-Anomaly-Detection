from app import app, db
from flask_migrate import Migrate, upgrade, init, migrate
from datetime import datetime, timezone
import os
import sqlite3

def add_columns_directly():
    """Add columns directly to the SQLite database as a fallback"""
    try:
        print("Attempting direct schema modification...")
        conn = sqlite3.connect('instance/users.db')
        cursor = conn.cursor()
        
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(user)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Add columns if they don't exist
        if 'allergies' not in columns:
            cursor.execute("ALTER TABLE user ADD COLUMN allergies TEXT")
            print("Added allergies column")
            
        if 'medications' not in columns:
            cursor.execute("ALTER TABLE user ADD COLUMN medications TEXT")
            print("Added medications column")
            
        if 'emergency_contact' not in columns:
            cursor.execute("ALTER TABLE user ADD COLUMN emergency_contact TEXT")
            print("Added emergency_contact column")
            
        if 'last_updated' not in columns:
            cursor.execute("ALTER TABLE user ADD COLUMN last_updated TIMESTAMP")
            print("Added last_updated column")
            
        conn.commit()
        conn.close()
        print("Direct schema modification completed successfully!")
        return True
    except Exception as e:
        print(f"Error during direct schema modification: {str(e)}")
        return False

def main():
    """Run database migrations to add new fields to User model"""
    print("Starting database migration...")
    
    try:
        # First try with Flask-Migrate
        # Initialize migration if not already done
        if not os.path.exists('migrations'):
            print("Initializing migration directory...")
            with app.app_context():
                init()
        
        # Create migration
        with app.app_context():
            print("Creating migration for database schema update...")
            migrate(message="Add health fields to User model")
        
        # Apply migration
        with app.app_context():
            print("Applying migration...")
            upgrade()
            
        print("Migration completed successfully using Flask-Migrate!")
    except Exception as e:
        print(f"Flask-Migrate error: {str(e)}")
        print("Falling back to direct database modification...")
        
        # Try direct SQLite modification as fallback
        if add_columns_directly():
            print("Successfully added columns via direct SQL.")
        else:
            print("Failed to update database schema. Please check error messages.")
            return
    
    print("\nDatabase update complete! New fields added to User model:")
    print("- allergies (text)")
    print("- medications (text)")
    print("- emergency_contact (text)")
    print("- last_updated (timestamp)")
    print("\nYou may need to restart the application for changes to take effect.")

if __name__ == "__main__":
    main() 