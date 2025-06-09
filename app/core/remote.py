import logging
from typing import Dict, Any, Optional, Union
import torch
from openfabric_pysdk.context import ExecutionResult

logger = logging.getLogger(__name__)

class Remote:
    """
    Remote is a helper class that interfaces with an Openfabric Proxy instance
    to send input data, execute computations, and fetch results synchronously
    or asynchronously.

    Attributes:
        host (str): The host to connect to.
        client (Optional[Proxy]): The initialized proxy client instance.
    """

    # ----------------------------------------------------------------------
    def __init__(self, host: str):
        """
        Initialize the Remote connection.
        
        Args:
            host (str): The host to connect to
        """
        self.host = host
        self.client = None
        self._verify_cuda()
        
    def _verify_cuda(self):
        """Verify CUDA availability and log device information."""
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Some operations may be slower.")
        else:
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # ----------------------------------------------------------------------
    def connect(self) -> bool:
        """
        Connect to the remote host.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Log memory usage before connection
            if torch.cuda.is_available():
                logger.info(f"Memory before connection: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
            # Connect to host
            # ... connection logic here ...
            
            # Log memory usage after connection
            if torch.cuda.is_available():
                logger.info(f"Memory after connection: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
            return True
        except Exception as e:
            logger.error(f"Error connecting to {self.host}: {str(e)}")
            return False

    # ----------------------------------------------------------------------
    def execute(self, inputs: Dict[str, Any], uid: str) -> Optional[ExecutionResult]:
        """
        Execute a request on the remote host.
        
        Args:
            inputs (Dict[str, Any]): The input parameters
            uid (str): The user ID
            
        Returns:
            Optional[ExecutionResult]: The execution result
        """
        if self.client is None:
            logger.error("Not connected to remote host")
            return None
            
        try:
            # Log memory usage before execution
            if torch.cuda.is_available():
                logger.info(f"Memory before execution: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
            # Execute request
            result = self.client.request(inputs, uid)
            
            # Log memory usage after execution
            if torch.cuda.is_available():
                logger.info(f"Memory after execution: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
            return result
        except Exception as e:
            logger.error(f"Error executing request: {str(e)}")
            return None

    # ----------------------------------------------------------------------
    def get_response(self, output: ExecutionResult) -> Optional[Dict[str, Any]]:
        """
        Get the response from an execution result.
        
        Args:
            output (ExecutionResult): The execution result
            
        Returns:
            Optional[Dict[str, Any]]: The response data
        """
        if output is None:
            return None
            
        try:
            # Log memory usage before getting response
            if torch.cuda.is_available():
                logger.info(f"Memory before getting response: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
            # Wait for result
            output.wait()
            status = str(output.status()).lower()
            
            if status == "completed":
                result = output.data()
                
                # Log memory usage after getting response
                if torch.cuda.is_available():
                    logger.info(f"Memory after getting response: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                
                return result
                
            if status in ("cancelled", "failed"):
                raise Exception("The request to the proxy app failed or was cancelled!")
                
            return None
        except Exception as e:
            logger.error(f"Error getting response: {str(e)}")
            return None

    # ----------------------------------------------------------------------
    def execute_sync(self, inputs: Dict[str, Any], configs: Dict[str, Any], uid: str) -> Optional[Dict[str, Any]]:
        """
        Execute a synchronous request.
        
        Args:
            inputs (Dict[str, Any]): The input parameters
            configs (Dict[str, Any]): The configuration parameters
            uid (str): The user ID
            
        Returns:
            Optional[Dict[str, Any]]: The response data
        """
        if self.client is None:
            logger.error("Not connected to remote host")
            return None
            
        try:
            # Log memory usage before sync execution
            if torch.cuda.is_available():
                logger.info(f"Memory before sync execution: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
            # Execute request
            output = self.client.execute(inputs, configs, uid)
            result = self.get_response(output)
            
            # Log memory usage after sync execution
            if torch.cuda.is_available():
                logger.info(f"Memory after sync execution: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            
            return result
        except Exception as e:
            logger.error(f"Error in sync execution: {str(e)}")
            return None
