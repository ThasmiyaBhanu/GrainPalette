import sqlite3
import datetime
from contextlib import contextmanager
import json

class RiceClassificationDB:
    """
    RiceClassificationDB provides an interface for managing users, predictions, sessions,
    and model statistics in a SQLite database for a rice classification application.
    Attributes:
        db_path (str): Path to the SQLite database file.
    Methods:
        __init__(db_path='rice_classification.db'):
            Initializes the database connection and ensures required tables exist.
        init_database():
            Initializes the database schema with tables for users, predictions, 
            sessions, and model statistics.
        get_connection():
            Context manager for obtaining a SQLite database connection with row 
            factory set for dict-like access.
        create_user(username, email=None):
            Creates a new user or returns the existing user ID if the username already exists.
        get_user(username):
            Retrieves user information by username.
        update_user_login(user_id):
            Updates the last login timestamp for a user.
        save_prediction(user_id, image_filename, predicted_class, confidence, 
                all_predictions, processing_time, image_size, model_version='v1.0'):
            Saves a prediction record to the database and updates the user's total prediction count.
        get_user_predictions(user_id, limit=50, offset=0):
            Retrieves a paginated list of predictions made by a specific user.
        get_prediction_stats(user_id=None, days=30):
            Retrieves prediction statistics (overall and by class) for a user or globally 
            over a specified period.
        get_all_predictions(limit=100, offset=0):
            Retrieves a paginated list of all predictions, including associated usernames.
        get_prediction_by_id(prediction_id):
            Retrieves a specific prediction by its ID, including associated username.
        get_user_by_id(user_id):
            Retrieves user information by user ID.
        get_all_users(limit=100, offset=0):
            Retrieves a paginated list of all users.
        get_top_users(limit=10):
            Retrieves the top users ranked by total prediction count.
        cleanup_old_data(days=90):
            Deletes prediction records older than the specified number of days.
        export_user_data(user_id):
            Exports all data related to a user, including user info and all predictions, 
            for backup or analysis.
    """
    def __init__(self, db_path='rice_classification.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize the database with required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    total_predictions INTEGER DEFAULT 0
                )
            ''')

            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    image_filename TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    all_predictions TEXT,  -- JSON string of all class probabilities
                    processing_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    image_size TEXT,
                    model_version TEXT DEFAULT 'v1.0',
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            # Sessions table for tracking user sessions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_end TIMESTAMP,
                    predictions_count INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')

            # Model performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE DEFAULT CURRENT_DATE,
                    total_predictions INTEGER DEFAULT 0,
                    avg_confidence REAL,
                    most_predicted_class TEXT,
                    least_predicted_class TEXT,
                    avg_processing_time REAL
                )
            ''')

            conn.commit()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access to rows
        try:
            yield conn
        finally:
            conn.close()

    def create_user(self, username, email=None):
        """Create a new user or return existing user ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    'INSERT INTO users (username, email) VALUES (?, ?)',
                    (username, email)
                )
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # User already exists, return existing ID
                cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
                return cursor.fetchone()[0]

    def get_user(self, username):
        """Get user information by username"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            return cursor.fetchone()

    def update_user_login(self, user_id):
        """Update user's last login time"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?',
                (user_id,)
            )
            conn.commit()
    
    def save_prediction(self, user_id, image_filename, predicted_class, confidence, 
                       all_predictions, processing_time, image_size, model_version='v1.0'):
        """Save a prediction to the database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions 
                (user_id, image_filename, predicted_class, confidence, all_predictions, 
                 processing_time, image_size, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, image_filename, predicted_class, confidence, 
                  json.dumps(all_predictions), processing_time, image_size, model_version))
            
            # Update user's total predictions count
            cursor.execute(
                'UPDATE users SET total_predictions = total_predictions + 1 WHERE id = ?',
                (user_id,)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_user_predictions(self, user_id, limit=50, offset=0):
        """Get user's prediction history with pagination"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM predictions 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            ''', (user_id, limit, offset))
            
            predictions = []
            for row in cursor.fetchall():
                pred = dict(row)
                pred['all_predictions'] = json.loads(pred['all_predictions'])
                predictions.append(pred)
            return predictions
    
    def get_prediction_stats(self, user_id=None, days=30):
        """Get prediction statistics for a user or globally"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            base_query = '''
                SELECT 
                    COUNT(*) as total_predictions,
                    AVG(confidence) as avg_confidence,
                    AVG(processing_time) as avg_processing_time,
                    predicted_class,
                    COUNT(*) as class_count
                FROM predictions 
                WHERE created_at >= datetime('now', '-{} days')
            '''.format(days)

            if user_id:
                cursor.execute(base_query + ' AND user_id = ? GROUP BY predicted_class', (user_id,))
            else:
                cursor.execute(base_query + ' GROUP BY predicted_class')

            results = cursor.fetchall()

            # Get overall stats
            if user_id:
                cursor.execute('''
                    SELECT COUNT(*) as total, AVG(confidence) as avg_conf, 
                           AVG(processing_time) as avg_time
                    FROM predictions 
                    WHERE user_id = ? AND created_at >= datetime('now', '-{} days')
                '''.format(days), (user_id,))
            else:
                cursor.execute('''
                    SELECT COUNT(*) as total, AVG(confidence) as avg_conf, 
                           AVG(processing_time) as avg_time
                    FROM predictions 
                    WHERE created_at >= datetime('now', '-{} days')
                '''.format(days))
            
            overall = cursor.fetchone()
            
            return {
                'overall': dict(overall) if overall else {},
                'by_class': [dict(row) for row in results]
            }
    
    def get_all_predictions(self, limit=100, offset=0):
        """Get all predictions with pagination"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT p.*, u.username 
                FROM predictions p
                LEFT JOIN users u ON p.user_id = u.id
                ORDER BY p.created_at DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            
            predictions = []
            for row in cursor.fetchall():
                pred = dict(row)
                pred['all_predictions'] = json.loads(pred['all_predictions'])
                predictions.append(pred)
            return predictions
    
    def get_prediction_by_id(self, prediction_id):
        """Get a specific prediction by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT p.*, u.username 
                FROM predictions p
                LEFT JOIN users u ON p.user_id = u.id
                WHERE p.id = ?
            ''', (prediction_id,))
            
            row = cursor.fetchone()
            if row:
                pred = dict(row)
                pred['all_predictions'] = json.loads(pred['all_predictions'])
                return pred
            return None
    
    def get_user_by_id(self, user_id):
        """Get user information by user ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_all_users(self, limit=100, offset=0):
        """
        Retrieve a paginated list of all users from the database.

        Args:
            limit (int, optional): The maximum number of users to return. Defaults to 100.
            offset (int, optional): The number of users to skip before starting to collect 
                the result set. Defaults to 0.

        Returns:
            list[dict]: A list of dictionaries, each representing a user record, ordered by 
            creation date descending.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM users 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_top_users(self, limit=10):
        """Get top users by prediction count"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT username, total_predictions, created_at, last_login
                FROM users 
                ORDER BY total_predictions DESC 
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_data(self, days=90):
        """Clean up old prediction data beyond specified days"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM predictions 
                WHERE created_at < datetime('now', '-{} days')
            '''.format(days))
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count
    
    def export_user_data(self, user_id):
        """Export all user data for backup/analysis"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get user info
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            user_data = dict(cursor.fetchone())
            
            # Get all predictions
            cursor.execute('''
                SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at
            ''', (user_id,))
            predictions = []
            for row in cursor.fetchall():
                pred = dict(row)
                pred['all_predictions'] = json.loads(pred['all_predictions'])
                predictions.append(pred)
            
            return {
                'user': user_data,
                'predictions': predictions,
                'export_date': datetime.datetime.now().isoformat()
            }

    def clear_user_predictions(self, user_id):
        """Delete all predictions for a specific user."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM predictions WHERE user_id = ?', (user_id,))
            cursor.execute('UPDATE users SET total_predictions = 0 WHERE id = ?', (user_id,))
            conn.commit()

