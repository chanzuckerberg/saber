import mrcfile, skimage
import numpy as np

# Try to import hyperspy for Material Science dataset
try:
    import hyperspy.api as hs
    hyperspy_available = True
except:
    hyperspy_available = False

def get_tomogram(run, 
                 voxel_size: float = 10, 
                 algorithm: str = 'wbp',
                 raise_error: bool = False):
    
    voxel_spacing_obj = run.get_voxel_spacing(voxel_size)

    if voxel_spacing_obj is None:
        # Query Avaiable Voxel Spacings
        availableVoxelSpacings = [tomo.voxel_size for tomo in run.voxel_spacings]

        # Report to the user which voxel spacings they can use 
        message = (f"[Warning] No tomogram found for {run.name} with voxel size {voxel_size} and tomogram type {algorithm}"
                   f"Available spacings are: {', '.join(map(str, availableVoxelSpacings))}" ) 
        if raise_error:
            raise ValueError(message)
        else:
            print(message)
            return None
    
    tomogram = voxel_spacing_obj.get_tomogram(algorithm)
    if tomogram is None:
        # Get available algorithms
        availableAlgorithms = [tomo.tomo_type for tomo in run.get_voxel_spacing(voxel_size).tomograms]
        
        # Report to the user which algorithms are available
        message = (f"[Warning] No tomogram found for {run.name} with voxel size {voxel_size} and tomogram type {algorithm}"
                   f"Available algorithms are: {', '.join(availableAlgorithms)}")
        if raise_error:
            raise ValueError(message)
        else:
            print(message)
            return None

    return tomogram.numpy()

def get_segmentation_array(run, 
                           voxel_spacing: float,
                           segmentation_name: str, 
                           session_id=None,
                           user_id=None,
                           raise_error = False):

    seg = run.get_segmentations(name=segmentation_name, 
                                session_id = session_id,
                                user_id = user_id,
                                voxel_size = voxel_spacing)

    # No Segmentations Are Available, Result in Error
    if len(seg) == 0:
        # Get all available segmentations with their metadata
        available_segs = run.get_segmentations(voxel_size=voxel_spacing)
        seg_info = [(s.name, s.user_id, s.session_id) for s in available_segs]
        
        # Format the information for display
        seg_details = [f"(name: {name}, user_id: {uid}, session_id: {sid})" 
                      for name, uid, sid in seg_info]
        
        message = ( f'\nNo segmentation found matching:\n'
                    f'  name: {segmentation_name}, user_id: {user_id}, session_id: {session_id}\n'
                    f'Available segmentations in {run.name} are:\n  ' + 
                    '\n  '.join(seg_details) )
        if raise_error:
            raise ValueError(message)
        else:
            print(message)
            return None

    # No Segmentations Are Available, Result in Error
    if len(seg) > 1:
        print(f'[Warning] More Than 1 Segmentation is Available for the Query Information. '
              f'Available Segmentations are: {seg} '
              f'Defaulting to Loading: {seg[0]}\n')
    seg = seg[0]

    return seg.numpy()

def get_coordinates(run,                     # CoPick run object containing the segmentation data
                    name: str,               # Name of the object or protein for which coordinates are being extracted
                    user_id: str,            # Identifier of the user that generated the picks
                    session_id: str = None,  # Identifier of the session that generated the picks
                    voxel_size: float = 10,  # Voxel size of the tomogram, used for scaling the coordinates
                    raise_error: bool = False):
                           
    # Retrieve the pick points associated with the specified object and user ID
    picks = run.get_picks(object_name=name, user_id=user_id, session_id=session_id)

    if len(picks) == 0:
        # Get all available segmentations with their metadata

        available_picks = run.get_picks()
        picks_info = [(s.pickable_object_name, s.user_id, s.session_id) for s in available_picks]
        
        # Format the information for display
        picks_details = [f"(name: {name}, user_id: {uid}, session_id: {sid})" 
                      for name, uid, sid in picks_info]
        
        message = ( f'\nNo picks found matching:\n'
                    f'  name: {name}, user_id: {user_id}, session_id: {session_id}\n'
                    f'Available picks are:\n  '  
                    + '\n  '.join(picks_details) )
        if raise_error:
            raise ValueError(message)
        else:
            print(message)
            return None
    elif len(picks) > 1:
        # Format pick information for display
        picks_info = [(p.pickable_object_name, p.user_id, p.session_id) for p in picks]
        picks_details = [f"(name: {name}, user_id: {uid}, session_id: {sid})" 
                        for name, uid, sid in picks_info]

        print(f'[Warning] More than 1 pick is available for the query information.'
              f'\nAvailable picks are:\n  ' + 
              '\n  '.join(picks_details) +
              f'\nDefaulting to loading:\n {picks[0]}\n')
    points = picks[0].points

    # Initialize an array to store the coordinates
    nPoints = len(picks[0].points)                      # Number of points retrieved
    coordinates = np.zeros([len(picks[0].points), 3])   # Create an empty array to hold the (z, y, x) coordinates

    # Iterate over all points and convert their locations to coordinates in voxel space
    for ii in range(nPoints):
        coordinates[ii,] = [points[ii].location.z / voxel_size,   # Scale z-coordinate by voxel size
                            points[ii].location.y / voxel_size,   # Scale y-coordinate by voxel size
                            points[ii].location.x / voxel_size]   # Scale x-coordinate by voxel size

    # Return the array of coordinates
    return coordinates

def read_micrograph(input: str):
    if input.endswith('.mrc'):
        with mrcfile.open(input, permissive=True) as mrc:
            data = mrc.data
            pixel_size = mrc.voxel_size.x
        return data, pixel_size
    elif input.endswith('.tif') or input.endswith('.tiff'):
        return skimage.io.imread(input)
    elif hyperspy_available and input.endswith('.hspy'):
        signal = hs.load(input)
        data = signal.data
        axes = signal.axes_manager
        pixel_size = axes[0].scale
        return data, pixel_size
    else:
        raise ValueError(f"Unsupported file type: {input}")