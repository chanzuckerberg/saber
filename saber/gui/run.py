from saber import cli_context
import rich_click as click

def run_gui(input: str):
    from saber.gui.base.zarr_gui import launch_gui
    launch_gui(input)

def run_web_gui(
    input: str, output: str, port: int, 
    host: str, debug: bool
    ):
    from saber.gui.web.main import launch_web
    launch_web(input, output, port, host, debug)

########################################################
# GUI Command
########################################################

@click.command(context_settings=cli_context)
@click.option('-i','--input', type=str, required=True, 
              help="Path to the input Zarr file.")
def gui(input: str):
    """
    Saber GUI for annotating SAM2 segmentations with custom classes.
    """
    run_gui(input)

########################################################
# Web GUI Command
########################################################

@click.command(context_settings=cli_context)
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Path to local input directory containing Zarr files')
@click.option('--output', '-o', type=click.Path(), default=None,
              help='Path to local output directory for saved annotations')
@click.option('--port', '-p', type=int, default=9090,
              help='Port to run the web server on')
@click.option('--host', type=str, default='localhost',
              help='Host to bind the server to (localhost for external access)')
@click.option('--debug', default=False, type=bool,
              help='Run in debug mode')
def web(input: str, output: str, port: int, host: str, 
        debug: bool):
    """
    SABER Annotation GUI Web Server
    
    Examples:
        # Basic usage
        python main.py --input /path/to/zarr/files --port 8080
        
        # With output directory
        python main.py --input /data --output /annotations
        
        # Remote access via SSH tunnel
        ssh -L 8080:localhost:8080 user@remote-server
        Then access at http://localhost:8080
    """
    run_web_gui(input, output, port, host, debug)