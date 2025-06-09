import sqlite3
from datetime import datetime
import json
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class MemorySystem:
    def __init__(self, db_path: str = "memory.db"):
        """
        Initialize the memory system.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database with required tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create generations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_prompt TEXT NOT NULL,
                    expanded_prompt TEXT,
                    image_path TEXT,
                    model_path TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def store_generation(self, 
                        original_prompt: str,
                        expanded_prompt: str,
                        image_path: Optional[str] = None,
                        model_path: Optional[str] = None,
                        metadata: Optional[Dict] = None) -> int:
        """
        Store a generation in the database.
        
        Args:
            original_prompt (str): The original user prompt
            expanded_prompt (str): The expanded prompt from LLM
            image_path (str, optional): Path to the generated image
            model_path (str, optional): Path to the generated 3D model
            metadata (dict, optional): Additional metadata
            
        Returns:
            int: ID of the stored generation
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO generations 
                (original_prompt, expanded_prompt, image_path, model_path, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                original_prompt,
                expanded_prompt,
                image_path,
                model_path,
                json.dumps(metadata) if metadata else None
            ))
            
            generation_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Stored generation with ID: {generation_id}")
            return generation_id
            
        except Exception as e:
            logger.error(f"Error storing generation: {str(e)}")
            raise

    def get_generation(self, generation_id: int) -> Optional[Dict]:
        """
        Retrieve a generation by ID.
        
        Args:
            generation_id (int): ID of the generation to retrieve
            
        Returns:
            dict: Generation data or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM generations WHERE id = ?
            ''', (generation_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'id': row[0],
                    'original_prompt': row[1],
                    'expanded_prompt': row[2],
                    'image_path': row[3],
                    'model_path': row[4],
                    'metadata': json.loads(row[5]) if row[5] else None,
                    'created_at': row[6]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving generation: {str(e)}")
            return None

    def search_generations(self, query: str) -> List[Dict]:
        """
        Search generations by prompt content.
        
        Args:
            query (str): Search query
            
        Returns:
            list: List of matching generations
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM generations 
                WHERE original_prompt LIKE ? OR expanded_prompt LIKE ?
                ORDER BY created_at DESC
            ''', (f'%{query}%', f'%{query}%'))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [{
                'id': row[0],
                'original_prompt': row[1],
                'expanded_prompt': row[2],
                'image_path': row[3],
                'model_path': row[4],
                'metadata': json.loads(row[5]) if row[5] else None,
                'created_at': row[6]
            } for row in rows]
            
        except Exception as e:
            logger.error(f"Error searching generations: {str(e)}")
            return [] 