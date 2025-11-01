"""
Command-line entry point for the SABER Annotation GUI Web Server
"""

from saber.gui.web.server import run_server
from saber import cli_context
import click, logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command(context_settings=cli_context,name='web')
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Path to local input directory containing Zarr files')
@click.option('--output', '-o', type=click.Path(), default=None,
              help='Path to local output directory for saved annotations')
@click.option('--port', '-p', type=int, default=9090,
              help='Port to run the web server on')
@click.option('--host', '-h', type=str, default='localhost',
              help='Host to bind the server to (localhost for external access)')
@click.option('--dask-scheduler', type=str, default=None,
              help='Dask scheduler address (e.g., tcp://localhost:8786)')
@click.option('--workers', '-w', type=int, default=4,
              help='Number of Dask workers to spawn (if no scheduler provided)')
@click.option('--debug/--no-debug', default=False,
              help='Run in debug mode')
def main(input, output, port, host, dask_scheduler, workers, debug):
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
    
    if dask_scheduler:
        logger.info(f"Using Dask scheduler at: {dask_scheduler}")
    else:
        logger.info(f"Starting local Dask cluster with {workers} workers")
    
    # Run the server
    run_server(
        data_path=input,
        output_path=output,
        host=host,
        port=port,
        dask_scheduler=dask_scheduler,
        n_workers=workers,
        debug=debug
    )

if __name__ == '__main__':
    main()