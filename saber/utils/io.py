import mrcfile, skimage, torch
from skimage import io as skio
import numpy as np

# Try to import hyperspy for Material Science dataset
try:
    import hyperspy.api as hs
    hyperspy_available = True
except:
    hyperspy_available = False

def read_micrograph(fname: str):
    """
    Read a micrograph from a file.
    Supports: MRC (.mrc), TIFF (.tif/.tiff), STEM (.dm4/.ser)
    Returns:
        data: np.ndarray
        pixel_size: float or None [Angstroms]
    """

    if fname.endswith('.mrc'):                 # MRC file
        with mrcfile.open(fname, permissive=True) as mrc:
            data = mrc.data
            pixel_size = mrc.voxel_size.x
        return data, pixel_size
    elif fname.endswith(('.tif', '.tiff')):     # TIFF file
        return skimage.io.imread(fname), None
    elif fname.endswith(('.dm4', '.ser')):     # STEM file
        if not hyperspy_available:
            raise ValueError("Hyperspy is not installed. Please install it to read .dm4 or .ser files. (pip install hyperspy)")
        return read_stem_micrograph(fname)

    # Unsupported file
    raise ValueError(f"Unsupported file type: {fname}")

def read_stem_micrograph(input: str):
    """
    Read a STEM micrograph from a file.
    Returns:
        data: np.ndarray
        pixel_size: float or None [Angstroms]
    """

    signal = hs.load(input)
    data = signal.data
    axes = signal.axes_manager
    pixel_size = axes[0].scale
    units = axes[0].units

    # Convert units to Angstroms
    if units == 'nm':
        pixel_size *= 10
    elif units == 'µm':
        pixel_size *= 1e3
    elif units == 'pm':
        pixel_size *= 1e-3
    else:
        raise ValueError(f"Unsupported unit: {units}")

    return data, pixel_size

def get_available_devices(deviceID: int = None):
    """
    Get the available devices for the current system.
    """
    # Set device
    if deviceID is None:
        if torch.cuda.is_available():           device_type = 'cuda'
        elif torch.backends.mps.is_available(): device_type = "mps" 
        else:                                   device_type = "cpu" 
        device = torch.device(device_type)
    else:
        device = determine_device(deviceID)
    return device

def determine_device(deviceID: int = 0):
    """
    Determine the device for the given deviceID.
    """

    # First check if CUDA is available at all
    if torch.cuda.is_available():
        try:

            # Make sure the device ID is valid
            device_count = torch.cuda.device_count()
            if deviceID >= device_count:
                print(f"Warning: Requested CUDA device {deviceID} but only {device_count} devices available")
                print(f"Falling back to device 0")
                deviceID = 0

            # Safely try to get the device properties
            props = torch.cuda.get_device_properties(deviceID)
            device = torch.device(f"cuda:{deviceID}")
            
            # Enable TF32 for Ampere GPUs if available
            if props.major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                # Only set up autocast after confirming device works
                # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            
        except Exception as e:
            print(f"Error accessing CUDA device {deviceID}: {e}")
            print("Falling back to CPU")
            device = torch.device("cpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    else:
        device = torch.device("cpu")
        print("Using CPU for computation (no GPU available)")

    return device


def mask3D_to_tiff(mask3D, output_path: str):
    """
    Convert a 3D mask to a TIFF file.
    """
    skio.imsave(output_path, mask3D)