import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import os

logger = logging.getLogger(__name__)

class MemorySystem:
    """
    Enhanced memory system with temporal queries and similarity search
    """
    
    def __init__(self, db_path: str = "generations.db"):
        self.db_path = db_path
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.stored_embeddings = []
        self.stored_ids = []
        
        # Initialize database
        self._init_db()
        # Load existing embeddings
        self._load_embeddings()
        
    def _init_db(self):
        """Initialize SQLite database with enhanced schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    original_prompt TEXT NOT NULL,
                    expanded_prompt TEXT,
                    image_path TEXT,
                    model_path TEXT,
                    cuda_device TEXT,
                    cuda_memory_used REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
            
    def _load_embeddings(self):
        """Load and index existing prompt embeddings"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all prompts
            cursor.execute('SELECT id, original_prompt, expanded_prompt FROM generations')
            rows = cursor.fetchall()
            
            if rows:
                # Combine original and expanded prompts for richer embedding
                texts = [f"{row[1]} {row[2]}" if row[2] else row[1] for row in rows]
                self.stored_ids = [row[0] for row in rows]
                
                # Generate embeddings
                self.stored_embeddings = self.encoder.encode(texts)
                
                # Initialize FAISS index
                dimension = self.stored_embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
                self.index.add(self.stored_embeddings)
                
            conn.close()
            logger.info(f"Loaded {len(self.stored_ids)} embeddings")
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {str(e)}")
            self.index = None
            
    def store_generation(
        self,
        user_id: str,
        original_prompt: str,
        expanded_prompt: str,
        image_path: str,
        model_path: str,
        metadata: Dict = None
    ) -> bool:
        """Store a new generation with enhanced metadata"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store generation
            cursor.execute('''
                INSERT INTO generations (
                    user_id, original_prompt, expanded_prompt, 
                    image_path, model_path, cuda_device,
                    cuda_memory_used, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, original_prompt, expanded_prompt,
                image_path, model_path,
                torch.cuda.current_device() if torch.cuda.is_available() else None,
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                json.dumps(metadata) if metadata else None
            ))
            
            generation_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            # Update embeddings
            if self.index is not None:
                text = f"{original_prompt} {expanded_prompt}"
                embedding = self.encoder.encode([text])[0]
                
                self.stored_ids.append(generation_id)
                if len(self.stored_embeddings) == 0:
                    self.stored_embeddings = embedding.reshape(1, -1)
                    self.index = faiss.IndexFlatL2(embedding.shape[0])
                else:
                    self.stored_embeddings = np.vstack([self.stored_embeddings, embedding])
                
                self.index.reset()
                self.index.add(self.stored_embeddings)
            
            logger.info(f"Stored generation {generation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store generation: {str(e)}")
            return False
            
    def find_similar_generations(self, prompt: str, k: int = 5) -> List[Dict]:
        """Find similar generations using FAISS similarity search"""
        try:
            if self.index is None or len(self.stored_ids) == 0:
                return []
                
            # Generate embedding for query
            query_embedding = self.encoder.encode([prompt])
            
            # Search
            distances, indices = self.index.search(query_embedding, k)
            
            # Get full generation data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx >= len(self.stored_ids):  # Safety check
                    continue
                    
                generation_id = self.stored_ids[idx]
                cursor.execute(
                    'SELECT * FROM generations WHERE id = ?',
                    (generation_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    results.append({
                        'id': row[0],
                        'user_id': row[1],
                        'original_prompt': row[2],
                        'expanded_prompt': row[3],
                        'image_path': row[4],
                        'model_path': row[5],
                        'similarity_score': float(1 / (1 + distance))
                    })
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar generations: {str(e)}")
            return []
            
    def find_by_date(self, date_str: str) -> List[Dict]:
        """Find generations by date description (e.g. 'last Thursday')"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert date string to datetime range
            target_date = self._parse_date_description(date_str)
            if not target_date:
                return []
            
            # Query within the target date
            cursor.execute('''
                SELECT * FROM generations 
                WHERE DATE(created_at) = DATE(?)
                ORDER BY created_at DESC
            ''', (target_date.strftime('%Y-%m-%d'),))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [{
                'id': row[0],
                'user_id': row[1],
                'original_prompt': row[2],
                'expanded_prompt': row[3],
                'image_path': row[4],
                'model_path': row[5],
                'created_at': row[8]
            } for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to find generations by date: {str(e)}")
            return []
            
    def _parse_date_description(self, date_str: str) -> Optional[datetime]:
        """Convert natural date description to datetime object"""
        try:
            from dateutil import parser
            from dateutil.relativedelta import relativedelta
            
            # Handle relative date descriptions
            date_str = date_str.lower()
            today = datetime.now()
            
            if 'last' in date_str:
                day_name = date_str.replace('last', '').strip()
                # Convert day name to datetime
                target = parser.parse(day_name)
                # Find the most recent occurrence of that day
                days_ago = (today.weekday() - target.weekday()) % 7
                if days_ago == 0:
                    days_ago = 7  # If today, get last week's
                return today - relativedelta(days=days_ago)
            else:
                # Try to parse as absolute date
                return parser.parse(date_str)
                
        except Exception as e:
            logger.error(f"Failed to parse date description: {str(e)}")
            return None
            
    def get_recent_generations(self, limit: int = 10) -> List[Dict]:
        """
        Get the most recent generations
        
        Args:
            limit (int): Maximum number of generations to return
            
        Returns:
            List[Dict]: List of recent generations with their metadata
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM generations 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [{
                'id': row[0],
                'user_id': row[1],
                'original_prompt': row[2],
                'expanded_prompt': row[3],
                'image_path': row[4],
                'model_path': row[5],
                'created_at': row[8],
                'metadata': json.loads(row[9]) if row[9] else None
            } for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get recent generations: {str(e)}")
            return []

    def get_generation_stats(self) -> Dict:
        """
        Get statistics about stored generations.
        
        Returns:
            Dict: Statistics about the stored generations
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute('SELECT COUNT(*) FROM generations')
            total_count = cursor.fetchone()[0]
            
            # Get average CUDA memory usage
            cursor.execute('SELECT AVG(cuda_memory_used) FROM generations')
            avg_memory = cursor.fetchone()[0] or 0
            
            # Get most used CUDA device
            cursor.execute('SELECT cuda_device, COUNT(*) FROM generations GROUP BY cuda_device ORDER BY COUNT(*) DESC LIMIT 1')
            most_used_device = cursor.fetchone()
            
            conn.close()
            
            stats = {
                'total_generations': total_count,
                'average_cuda_memory_used_mb': round(avg_memory, 2),
                'most_used_device': most_used_device[0] if most_used_device else 'None'
            }
            
            logger.info(f"Retrieved generation stats: {stats}")
            return stats
        except Exception as e:
            logger.error(f"Error getting generation stats: {str(e)}")
            return {
                'total_generations': 0,
                'average_cuda_memory_used_mb': 0,
                'most_used_device': 'None'
            } 