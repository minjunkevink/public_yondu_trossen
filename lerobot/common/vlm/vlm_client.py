import requests
import time
import logging
from typing import Optional, Dict, Any, Tuple
import json
from threading import Thread, Lock
from queue import Queue
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)

class VLMClient:
    def __init__(
        self,
        api_url: str,
        api_key: str,
        query_interval_s: float = 1.0,
        max_retries: int = 3,
        timeout_s: float = 5.0,
        max_queue_size: int = 5,
        response_timeout_s: float = 2.0
    ):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.query_interval_s = query_interval_s
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.max_queue_size = max_queue_size
        self.response_timeout_s = response_timeout_s
        
        self.last_query_time = 0
        self.request_queue = Queue(maxsize=max_queue_size)
        self.response_queue = Queue(maxsize=max_queue_size)
        self.lock = Lock()
        
        # Start the worker thread
        self.worker_thread = Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        # Track latency statistics
        self.latency_history = []
        self.max_history_size = 100

    def _worker_loop(self):
        """Background worker that processes VLM requests asynchronously."""
        while True:
            try:
                # Get request from queue
                request_data = self.request_queue.get()
                if request_data is None:  # Shutdown signal
                    break
                    
                image_data, prompt, timestamp = request_data
                
                # Process request
                response = self._make_request(image_data, prompt)
                
                # Calculate latency
                latency = time.time() - timestamp
                with self.lock:
                    self.latency_history.append(latency)
                    if len(self.latency_history) > self.max_history_size:
                        self.latency_history.pop(0)
                
                # Put response in queue
                self.response_queue.put((response, timestamp))
                
            except Exception as e:
                logger.error(f"Error in VLM worker thread: {str(e)}")
                time.sleep(0.1)  # Prevent tight loop on error

    def _make_request(self, image_data: bytes, prompt: str) -> Optional[Dict[str, Any]]:
        """Make the actual HTTP request to the VLM API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        files = {
            'image': ('image.jpg', image_data, 'image/jpeg')
        }
        data = {
            'prompt': prompt
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_url}/predict",
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=self.timeout_s
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"VLM query attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    logger.error("All VLM query attempts failed")
                    return None
                time.sleep(1)
        
        return None

    def query_vlm(self, image_data: bytes, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Query the VLM API with an image and prompt.
        Uses asynchronous processing to handle latency.
        
        Args:
            image_data: The image data in bytes
            prompt: The prompt to send to the VLM
            
        Returns:
            Optional[Dict[str, Any]]: The VLM response or None if the query failed
        """
        # Ensure we don't query too frequently
        current_time = time.time()
        if current_time - self.last_query_time < self.query_interval_s:
            time.sleep(self.query_interval_s - (current_time - self.last_query_time))
        
        # Try to put request in queue
        try:
            self.request_queue.put_nowait((image_data, prompt, current_time))
            self.last_query_time = current_time
        except Queue.Full:
            logger.warning("VLM request queue full, dropping request")
            return None
        
        # Wait for response with timeout
        try:
            response, timestamp = self.response_queue.get(timeout=self.response_timeout_s)
            return response
        except Queue.Empty:
            logger.warning("Timeout waiting for VLM response")
            return None

    def get_latency_stats(self) -> Tuple[float, float, float]:
        """
        Get statistics about VLM response latency.
        
        Returns:
            Tuple[float, float, float]: (mean, std, max) latency in seconds
        """
        with self.lock:
            if not self.latency_history:
                return 0.0, 0.0, 0.0
            latencies = np.array(self.latency_history)
            return float(np.mean(latencies)), float(np.std(latencies)), float(np.max(latencies))

    def process_vlm_response(self, response: Dict[str, Any]) -> Optional[str]:
        """
        Process the VLM response and extract the relevant information.
        
        Args:
            response: The VLM API response
            
        Returns:
            Optional[str]: The processed response or None if processing failed
        """
        try:
            if not response:
                return None
                
            # Extract the relevant information from the response
            if 'output' in response:
                return response['output']
            elif 'text' in response:
                return response['text']
            else:
                logger.warning(f"Unexpected VLM response format: {response}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing VLM response: {str(e)}")
            return None

    def shutdown(self):
        """Clean shutdown of the VLM client."""
        self.request_queue.put(None)  # Signal worker thread to stop
        self.worker_thread.join(timeout=5.0) 