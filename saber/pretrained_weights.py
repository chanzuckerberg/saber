import shutil, click, sys, os, subprocess, saber

@click.group(name="download")
@click.pass_context
def cli(ctx):
    """Download the pretrained weights of SAM 2.1 and MemBrain."""
    pass


@cli.command(context_settings={"show_default": True})
def sam2_weights():
    download_sam2_weights()

def download_sam2_weights():
    """
    Downloads SAM 2.1 checkpoints using either wget or curl.
    """
    # Create the download directory if it does not exist.
    download_dir = os.path.join(os.path.dirname(saber.__file__), 'checkpoints')
    os.makedirs(download_dir, exist_ok=True)

    # Check for wget or curl availability.
    if shutil.which("wget"):
        download_tool = "wget"
        use_wget = True
    elif shutil.which("curl"):
        download_tool = "curl"
        use_wget = False
    else:
        print("Please install wget or curl to download the checkpoints.")
        sys.exit(1)

    # Define the base URL and the SAM 2.1 checkpoints.
    # sam2_base_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824"
    sam2_1_base_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
    checkpoints = {
        "sam2.1_hiera_tiny.pt": f"{sam2_1_base_url}/sam2.1_hiera_tiny.pt",
        "sam2.1_hiera_small.pt": f"{sam2_1_base_url}/sam2.1_hiera_small.pt",
        "sam2.1_hiera_base_plus.pt": f"{sam2_1_base_url}/sam2.1_hiera_base_plus.pt",
        "sam2.1_hiera_large.pt": f"{sam2_1_base_url}/sam2.1_hiera_large.pt",
    }

    # Download each checkpoint.
    for filename, url in checkpoints.items():
        print(f"Downloading {filename} checkpoint...")
        if use_wget:
            # For wget, use the -P option to specify the download directory.
            cmd = [download_tool, "-P", download_dir, url]
        else:
            # For curl, specify the output file with -o.
            output_file = os.path.join(download_dir, filename)
            cmd = [download_tool, "-L", url, "-o", output_file]
        
        result = subprocess.call(cmd)
        if result != 0:
            print(f"Failed to download checkpoint from {url}")
            sys.exit(1)

    print("All checkpoints are downloaded successfully.") 
    
def get_sam2_checkpoint(sam2_cfg: str):
    """
    Get the checkpoint path for the SAM 2.1 model based on the provided configuration.
    """
    
    # Determine the directory where checkpoint files are stored
    checkpoint_dir = os.path.join(os.path.dirname(saber.__file__), 'checkpoints')

    # Dictionary mapping each configuration to a tuple of (config file, checkpoint path)
    config_map = {
        'large': ('sam2.1_hiera_l.yaml', os.path.join(checkpoint_dir, 'sam2.1_hiera_large.pt')),
        'base':  ('sam2.1_hiera_b+.yaml', os.path.join(checkpoint_dir, 'sam2.1_hiera_base_plus.pt')),
        'small': ('sam2.1_hiera_s.yaml', os.path.join(checkpoint_dir, 'sam2.1_hiera_small.pt')),
        'tiny':  ('sam2.1_hiera_t.yaml', os.path.join(checkpoint_dir, 'sam2.1_hiera_tiny.pt'))
    }

    # Try to fetch the configuration values based on the provided sam2_cfg string.
    try:
        cfg, checkpoint = config_map[sam2_cfg]
    except KeyError:
        # If sam2_cfg is not a valid key in the dictionary, raise a ValueError with an informative message.
        raise ValueError(f'Invalid SAM2 Model Config: {sam2_cfg}')

    # Ensure the checkpoint file exists, if not, download the weights.
    if not os.path.exists(checkpoint):
        download_sam2_weights()

    # Return the full path to the configuration file and the checkpoint file.
    return os.path.join('configs/sam2.1', cfg), checkpoint

