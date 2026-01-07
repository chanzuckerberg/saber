"""
Command-line entry point for the SABER Annotation GUI Web Server
"""

from saber.gui.web.server import run_server
from saber import cli_context
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def launch_web(input, output, port, host, debug):
    """
    SABER Annotation GUI Web Server
    
    Examples:
        # Basic usage
        python main.py --input /path/to/zarr/files --port 8080
        
        # With output directory
        python main.py --input /data --output /annotations
        
        # With external Dask cluster
        python main.py --input /data --dask-scheduler tcp://scheduler:8786
        
        # Remote access via SSH tunnel
        ssh -L 8080:localhost:8080 user@remote-server
        Then access at http://localhost:8080
    """
    
    logger.info(f"Starting SABER Annotation GUI Server...")
    logger.info(f"Data directory: {input}")
    logger.info(f"Output directory: {output or 'Current directory'}")
    logger.info(f"Server: http://{host}:{port}")
    
    # Run the server
    run_server(
        data_path=input,
        output_path=output,
        host=host,
        port=port,
        debug=debug
    )
