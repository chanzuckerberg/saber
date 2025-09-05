from skimage.measure import regionprops
from copick_utils.io import writers
import numpy as np

def extract_organelle_statistics(
    run, mask, organelle_name, session_id, user_id, 
    voxel_size, save_copick = True, save_statistics=True, xyz_order=True):
    """
    Extract statistics and return CSV rows.
    
    Returns:
        List of CSV rows if save_statistics is True, empty list otherwise
    """

    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]  # Ignore background (label 0)

    coordinates = {}
    csv_rows = []
    for label in unique_labels:
        
        component_mask = (mask == label).astype("int")
        rprops = regionprops(component_mask)[0]
        centroid = rprops.centroid
        
        # Flip Coordinates to X, Y, Z Order
        if xyz_order:
            centroid = centroid[::-1]
        coordinates[str(label)] = centroid
        
        if save_statistics:
            
            # Compute Volume in nm^3
            volume = np.sum(component_mask) * (voxel_size/10)**3 # Convert from Angstom to nm^3

            # Sort axes to identify the first (Z-biased) and two in-plane dimensions
            axes_lengths = sorted([rprops.axis_major_length, rprops.axis_minor_length, 
                                   rprops.axis_minor_length])

            # Convert to physical units (nm)
            axis_x = axes_lengths[1] * (voxel_size/10)  # Likely an in-plane axis
            axis_y = axes_lengths[2] * (voxel_size/10)  # Likely an in-plane axis
            diameter = (axis_x + axis_y) / 2

            # Prepare row for CSV
            csv_row = [
                run.name,
                int(label),
                volume,
                diameter,
            ]
            csv_rows.append(csv_row)

    # Save Statistics to CSV File
    if len(coordinates) > 0:
        # Save Coordinates to Copick
        if save_copick:
            save_coordinates_to_copick(run, coordinates, organelle_name, 
                                      session_id, user_id, voxel_size)
    else:
        print(f"{run.name} didn't have any organelles present!")

    return csv_rows

def save_coordinates_to_copick(run, coordinates, organelle_name, session_id, user_id, voxel_size):

    # Assign Identity As Orientation
    orientations = np.zeros([len(coordinates), 4, 4])
    orientations[:,:3,:3] = np.identity(3)
    orientations[:,3,3] = 1

    # Extract the coordinate tuples and convert them into a numpy array
    points = np.array(list(coordinates.values()))
    points *= voxel_size

    # Check to see if the pickable object already exists, if not, create it
    try:
        picks = run.new_picks(object_name = organelle_name, 
                            session_id = session_id, 
                            user_id = user_id)

        picks.from_numpy( points, orientations )
    except Exception as e:
        print(f"Error creating picks for {run.name}: {e}")
