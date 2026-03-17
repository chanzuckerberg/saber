import shutil, click, sys, os, subprocess, saber

@click.group(name="download")
@click.pass_context
def cli(ctx):
    """Download the pretrained weights of SAM 2.1, SAM 3, and MemBrain."""
    pass


@cli.command(context_settings={"show_default": True})
def sam2_weights():
    download_sam2_weights()


@cli.command(context_settings={"show_default": True})
def sam3_weights():
    """Download SAM 3 checkpoint from HuggingFace (facebook/sam3)."""
    download_sam3_weights()

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


def download_sam3_weights():
    """
    Download the SAM 3 checkpoint from HuggingFace (facebook/sam3).

    Requires either:
      - A prior `huggingface-cli login`, or
      - The HF_TOKEN environment variable set to a valid token.

    The checkpoint is cached in HuggingFace's default cache directory
    (~/.cache/huggingface/hub) and reused on subsequent calls.
    """
    try:
        from sam3.model_builder import download_ckpt_from_hf
    except ImportError:
        print(
            "sam3 is not installed.  Install it with:\n"
            "  pip install git+https://github.com/facebookresearch/sam3"
        )
        sys.exit(1)

    print("Downloading SAM 3 checkpoint from HuggingFace (facebook/sam3) ...")
    try:
        checkpoint_path = download_ckpt_from_hf()
        dest = os.path.join(os.path.dirname(saber.__file__), "checkpoints", "sam3.pt")
        shutil.copy2(checkpoint_path, dest)
        print(f"SAM 3 checkpoint saved to: {dest}")
    except Exception as e:
        print(
            f"Download failed: {e}\n\n"
            "To download SAM3 weights you must first request access:\n"
            "  1. Go to https://huggingface.co/facebook/sam3 and request access\n"
            "  2. Log in:  huggingface-cli login\n"
        )
        sys.exit(1)


def get_sam3_bpe_path() -> str:
    """
    Return a valid path to the SAM3 BPE vocabulary file.

    Resolution order:
      1. saber's own checkpoints directory (cached after first download)
      2. sam3 package's assets directory (works when installed from source
         with assets bundled correctly)
      3. Download from OpenAI's public CDN (no authentication required;
         this is the original CLIP vocabulary file)

    The file is ~1 MB and is cached permanently after the first download.
    """
    BPE_FILENAME = "bpe_simple_vocab_16e6.txt.gz"
    BPE_URL = "https://openaipublic.azureedge.net/clip/bpe_simple_vocab_16e6.txt.gz"

    # 1. Check saber's checkpoints directory first
    checkpoint_dir = os.path.join(os.path.dirname(saber.__file__), "checkpoints")
    cached_path = os.path.join(checkpoint_dir, BPE_FILENAME)
    if os.path.exists(cached_path):
        return cached_path

    # 2. Try the sam3 package's own assets directory
    try:
        import pkg_resources
        pkg_path = pkg_resources.resource_filename("sam3", f"assets/{BPE_FILENAME}")
        if os.path.exists(pkg_path):
            return pkg_path
    except Exception:
        pass

    # 3. Download from OpenAI's public CDN and cache in saber's checkpoints
    print(f"BPE vocabulary not found locally — downloading from OpenAI CDN ...")
    os.makedirs(checkpoint_dir, exist_ok=True)
    try:
        import urllib.request
        urllib.request.urlretrieve(BPE_URL, cached_path)
        print(f"BPE vocabulary cached at: {cached_path}")
        return cached_path
    except Exception as e:
        raise RuntimeError(
            f"Could not download BPE vocabulary: {e}\n"
            f"Download it manually from:\n  {BPE_URL}\n"
            f"and pass the path as bpe_path= to build_sam3_image_model()."
        ) from e


def get_sam3_checkpoint():
    """
    Return the path to the cached SAM 3 checkpoint.

    Resolution order:
      1. saber/checkpoints/sam3.pt (copied here by download_sam3_weights)
      2. HuggingFace hub cache (~/.cache/huggingface/hub)

    Returns None if the checkpoint is not available locally.
    """
    # 1. Check saber's local checkpoints directory first
    local = os.path.join(os.path.dirname(saber.__file__), "checkpoints", "sam3.pt")
    if os.path.exists(local):
        return local

    # 2. Fall back to HF cache
    try:
        from huggingface_hub import try_to_load_from_cache
        return try_to_load_from_cache(repo_id="facebook/sam3", filename="sam3.pt")
    except Exception:
        return None


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

