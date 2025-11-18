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
        
        # Skip very small regions that might cause numerical issues
        if np.sum(component_mask) < 3:
            print(f"Skipping label {label} in {run.name}: too small (< 3 voxels)")
            continue
            
        try:
            rprops = regionprops(component_mask)[0]
            centroid = rprops.centroid
            
            # Flip Coordinates to X, Y, Z Order
            if xyz_order:
                centroid = centroid[::-1]
            coordinates[str(label)] = centroid
            
            if save_statistics:
                # Compute Volume in nm^3
                volume = np.sum(component_mask) * (voxel_size/10)**3 # Convert from Angstrom to nm^3

                # Try to get axis lengths with error handling
                try:
                    axis_major = rprops.axis_major_length
                    axis_minor = rprops.axis_minor_length
                    
                    # For 3D, use both axes (major and minor are the only ones available)
                    axis_x = axis_minor * (voxel_size/10)
                    axis_y = axis_major * (voxel_size/10)
                    diameter = (axis_x + axis_y) / 2
                    
                except (ValueError, AttributeError) as e:
                    # Fall back to equivalent sphere diameter if axis calculation fails
                    print(f"Warning: Could not compute axes for label {label} in {run.name}, using spherical approximation")
                    diameter = 2 * ((3 * volume) / (4 * np.pi)) ** (1/3)

                # Prepare row for CSV
                csv_row = [
                    run.name,
                    int(label),
                    volume,
                    diameter,
                ]
                csv_rows.append(csv_row)
                
        except (ValueError, IndexError) as e:
            print(f"Error processing label {label} in {run.name}: {e}")
            continue

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
