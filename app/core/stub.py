import json
import logging
import pprint
from typing import Any, Dict, List, Literal, Tuple, Optional

import requests
import torch
from openfabric_pysdk import Stub as BaseStub
from openfabric_pysdk.context import State

from core.remote import Remote
from openfabric_pysdk.helper import has_resource_fields, json_schema_to_marshmallow, resolve_resources
from openfabric_pysdk.loader import OutputSchemaInst

# Type aliases for clarity
Manifests = Dict[str, dict]
Schemas = Dict[str, Tuple[dict, dict]]
Connections = Dict[str, Remote]

logger = logging.getLogger(__name__)

class Stub(BaseStub):
    """
    Stub acts as a lightweight client interface that initializes remote connections
    to multiple Openfabric applications, fetching their manifests, schemas, and enabling
    execution of calls to these apps.

    Attributes:
        _schema (Schemas): Stores input/output schemas for each app ID.
        _manifest (Manifests): Stores manifest metadata for each app ID.
        _connections (Connections): Stores active Remote connections for each app ID.
    """

    # ----------------------------------------------------------------------
    def __init__(self, app_ids: List[str]):
        """
        Initializes the Stub instance by loading manifests, schemas, and connections
        for each given app ID.

        Args:
            app_ids (List[str]): A list of application identifiers (hostnames or URLs).
        """
        super().__init__(app_ids)
        self._verify_cuda()
        self._schema: Schemas = {}
        self._manifest: Manifests = {}
        self._connections: Connections = {}

        for app_id in app_ids:
            base_url = app_id.strip('/')

            try:
                # Fetch manifest
                manifest = requests.get(f"https://{base_url}/manifest", timeout=5).json()
                logging.info(f"[{app_id}] Manifest loaded: {manifest}")
                self._manifest[app_id] = manifest

                # Fetch input schema
                input_schema = requests.get(f"https://{base_url}/schema?type=input", timeout=5).json()
                logging.info(f"[{app_id}] Input schema loaded: {input_schema}")

                # Fetch output schema
                output_schema = requests.get(f"https://{base_url}/schema?type=output", timeout=5).json()
                logging.info(f"[{app_id}] Output schema loaded: {output_schema}")
                self._schema[app_id] = (input_schema, output_schema)

                # Establish Remote WebSocket connection
                self._connections[app_id] = Remote(f"wss://{base_url}/app", f"{app_id}-proxy").connect()
                logging.info(f"[{app_id}] Connection established.")
            except Exception as e:
                logging.error(f"[{app_id}] Initialization failed: {e}")

    # ----------------------------------------------------------------------
    def _verify_cuda(self):
        """Verify CUDA availability and log device information."""
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Some operations may be slower.")
        else:
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # ----------------------------------------------------------------------
    def call(self, app_id: str, inputs: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Call an app with the given inputs.
        
        Args:
            app_id (str): The ID of the app to call
            inputs (Dict[str, Any]): The input parameters
            user_id (str): The user ID
            
        Returns:
            Dict[str, Any]: The app's response
        """
        try:
            # Log memory usage before call
            if torch.cuda.is_available():
                logger.info(f"Memory before call: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
            # Make the call
            result = super().call(app_id, inputs, user_id)
            
            # Log memory usage after call
            if torch.cuda.is_available():
                logger.info(f"Memory after call: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
            return result
        except Exception as e:
            logger.error(f"Error calling app {app_id}: {str(e)}")
            raise
    
    # ----------------------------------------------------------------------
    def execute(self, app_id: str, inputs: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """
        Execute an app with the given inputs.
        
        Args:
            app_id (str): The ID of the app to execute
            inputs (Dict[str, Any]): The input parameters
            user_id (str): The user ID
            
        Returns:
            Dict[str, Any]: The app's response
        """
        try:
            # Log memory usage before execution
            if torch.cuda.is_available():
                logger.info(f"Memory before execution: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
            # Execute the app
            result = super().execute(app_id, inputs, user_id)
            
            # Log memory usage after execution
            if torch.cuda.is_available():
                logger.info(f"Memory after execution: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
            return result
        except Exception as e:
            logger.error(f"Error executing app {app_id}: {str(e)}")
            raise
    
    # ----------------------------------------------------------------------
    def get_state(self) -> Optional[State]:
        """
        Get the current state.
        
        Returns:
            Optional[State]: The current state
        """
        try:
            state = super().get_state()
            if state and torch.cuda.is_available():
                logger.info(f"Memory in state: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            return state
        except Exception as e:
            logger.error(f"Error getting state: {str(e)}")
            return None
    
    # ----------------------------------------------------------------------
    def set_state(self, state: State) -> None:
        """
        Set the current state.
        
        Args:
            state (State): The state to set
        """
        try:
            super().set_state(state)
            if torch.cuda.is_available():
                logger.info(f"Memory after state update: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        except Exception as e:
            logger.error(f"Error setting state: {str(e)}")
            raise

    # ----------------------------------------------------------------------
    def manifest(self, app_id: str) -> dict:
        """
        Retrieves the manifest metadata for a specific application.

        Args:
            app_id (str): The application ID for which to retrieve the manifest.

        Returns:
            dict: The manifest data for the app, or an empty dictionary if not found.
        """
        return self._manifest.get(app_id, {})

    # ----------------------------------------------------------------------
    def schema(self, app_id: str, type: Literal['input', 'output']) -> dict:
        """
        Retrieves the input or output schema for a specific application.

        Args:
            app_id (str): The application ID for which to retrieve the schema.
            type (Literal['input', 'output']): The type of schema to retrieve.

        Returns:
            dict: The requested schema (input or output).

        Raises:
            ValueError: If the schema type is invalid or the schema is not found.
        """
        _input, _output = self._schema.get(app_id, (None, None))

        if type == 'input':
            if _input is None:
                raise ValueError(f"Input schema not found for app ID: {app_id}")
            return _input
        elif type == 'output':
            if _output is None:
                raise ValueError(f"Output schema not found for app ID: {app_id}")
            return _output
        else:
            raise ValueError("Type must be either 'input' or 'output'")
