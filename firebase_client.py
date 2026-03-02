"""
Firebase Client for AETA State Management
Implements robust Firebase integration with retry logic and error handling
"""
import os
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import time
from contextlib import contextmanager
import structlog

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, db
    from google.api_core.exceptions import GoogleAPIError, DeadlineExceeded
    from google.cloud.firestore_v1 import DocumentReference, CollectionReference
except ImportError as e:
    raise ImportError("firebase-admin not installed. Run: pip install firebase-admin") from e

logger = structlog.get_logger(__name__)

class FirebaseClient:
    """Firebase client with retry logic and connection management"""
    
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    
    def __init__(self, config):
        self.config = config
        self._app = None
        self._fs_client = None
        self._rtdb_client = None
        self._initialized = False
        self._initialize_firebase()
    
    def _initialize_firebase(self) -> None:
        """Initialize Firebase with proper error handling"""
        try:
            if firebase_admin._apps:
                self._app = firebase_admin.get_app()
                logger.info("Using existing Firebase app")
            else:
                creds_path = self.config.firebase.credentials_path
                if not os.path.exists(creds_path):
                    raise FileNotFoundError(f"Firebase credentials not found at: {creds_path}")
                
                cred = credentials.Certificate(creds_path)
                self._app = firebase_admin.initialize_app(
                    cred,
                    {
                        'projectId': self.config.firebase.project_id,
                        'databaseURL': f"https://{self.config.firebase.project_id}.firebaseio.com"
                    }
                )
                logger.info("Firebase app initialized successfully")