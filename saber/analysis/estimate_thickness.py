from saber.visualization import sam2 as viz
from scipy.optimize import curve_fit
import numpy as np
import zarr, os

# Define the Function to Fit
def quadratic(x, a, b, c, d):
    return d * np.maximum(a * (x - b)**2 + c , 0)    

def fit_quadratic(x, data):
    nFrames = data.shape[0]
    x_max = np.argmax(data[1:-1])
    popt1, _ = curve_fit(
        quadratic, x, data, 
        p0=[-1e-3, x_max, 1, np.max(data)/2], 
        bounds=([-np.inf, 0, 0, 0  ], 
                [0, nFrames, 10, 10]) )

    r2_quad = calculate_r2_score(data, quadratic, popt1)

    return popt1, r2_quad

def gaussian(x, a, b, c):
    with np.errstate(over='ignore'):
        return a * np.exp(-(x-b)**2 / (2 * c**2))

def fit_gaussian(x, data):

    # Estimate Max Bound 
    nFrames = data.shape[0]
    x_max = np.argmax(data[1:-1])
    c_max = nFrames * 0.25 / 2.355

    # Option 2 - Gaussian Fit and Calculate R2 Score
    popt2, _ = curve_fit(
        gaussian, x, data,  
        p0=[np.max(data), x_max, 3e-1 ], 
        bounds=((0, 0, 0), (np.inf, nFrames, c_max)) 
    )
    r2_gauss = calculate_r2_score(data, gaussian, popt2)

    return popt2, r2_gauss

def calculate_r2_score(data, func, fit_params):
    x = np.arange(len(data))
    y_fit = func(x, *fit_params)
    ss_res = np.sum((data - y_fit) ** 2)
    ss_tot = np.sum((data - np.mean(data)) ** 2)

    # If the Total Sum of Squares is 0, Return 0
    if ss_tot == 0: r2 = 0
    else: r2 = 1 - ss_res / ss_tot
    return r2

def preprocess(data: np.ndarray):

    # Ensure the Data is Non-Negative
    data = np.maximum(data, 0)

    # Subtract the Mean of the Last 10 Frames
    data -= np.mean(data[-15:-5])

    # Ensure the Data is Non-Negative
    data = np.maximum(data, 0)

    return data

def fit_organelle_boundaries(frame_scores: np.ndarray, plot: bool = False):
    """
    Fit boundaries for organelles based on frame scores.
    """

    (nFrames, nMasks) = frame_scores.shape
    mask_boundaries = np.zeros((nFrames, nMasks))
    for ii in range(nMasks):

        # Get the Scores for the Current Mask
        data = frame_scores[:,ii].copy()
        
        # Preprocess the Data
        data = preprocess(data)

        # Create x values (indices)
        x = np.arange(len(data), dtype=np.float32)       

        # Fit the function to the data
        try:
            popt1, r2_quad = fit_quadratic(x, data)
        except Exception as e:
            print(f"Error fitting Quadratic mask {ii}: {e}")
            r2_quad = 0

        try: 
            popt2, r2_gauss = fit_gaussian(x, data)
        except Exception as e:
            print(f'Error fitting Gaussian mask {ii}: {e}')
            r2_gauss = 0

        if r2_quad == 0 and r2_gauss == 0:
            mask_boundaries[:,ii] = np.zeros(data.shape[0])
        elif r2_quad > r2_gauss:
            func = quadratic; parameters = popt1
            mask_boundaries[:,ii] = quadratic(x, *popt1)
        else:
            func = gaussian; parameters = popt2
            mask_boundaries[:,ii] = gaussian(x, *popt2)

        # Optional - Plot the Fit
        if plot:
            viz.plot_fit(data, func, parameters)   

    return mask_boundaries

def save_frame_scores(run, frame_scores):
    # Save frame scores to zarr file
    
    # Create zarr file if it doesn't exist or open it if it does
    zarr_path = 'frame_scores.zarr'
    if os.path.exists(zarr_path):
        z = zarr.open(zarr_path, mode='a')
    else:
        z = zarr.open(zarr_path, mode='w')
    
    # Create the dataset with frame scores data
    z.create_dataset( run.name, data=frame_scores, overwrite=True)
    print(f"Saved frame scores as dataset: {zarr_path}")