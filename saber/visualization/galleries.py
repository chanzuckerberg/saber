import saber.visualization.classifier as viz
import time, math, glob, os, zarr
import matplotlib.pyplot as plt
from tqdm import tqdm

def initialize_page(figsize=(12, 12), dpi=150):
    """
    Initialize a figure with a grid of subplots.
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)

    plt.subplots_adjust(
        left=0.05,      # Left margin (smaller = less white space) 
        right=0.95,     # Right margin (larger = less white space)
        top=0.95,       # Top margin (larger = less white space)
        bottom=0.05,    # Bottom margin (smaller = less white space)
        wspace=0.05,    # Width spacing between subplots (smaller = less white space)
        hspace=0.1     # Height spacing between subplots (smaller = less white space)
    )
    return fig


def turn_off_unused_subplots(
    fig,
    rows,
    columns,
    page_images
    ):
    """
    Turn off any unused subplots.
    """
    # Turn off any unused subplots
    for i in range(page_images, rows * columns):
        r, c = divmod(i, columns)
        if r < rows and c < columns:
            subplot_idx = r * columns + c + 1
            ax = fig.add_subplot(rows, columns, subplot_idx)
            ax.axis('off')


def create_png_gallery(
    input_folder_path, 
    output_folder_path: str = None,
    rows=4, columns=4, dpi=150, 
    ):
    """
    Create a minimal PNG gallery with one PNG per page.
    
    Args:
        folder_path: Path to the folder containing PNG images
        output_filename: Base name of the output PNG files (no extension)
        rows: Number of rows in the gallery
        columns: Number of columns in the gallery
    """

    # Initialize Output Filename
    output_filename = "gallery_p"
    if output_folder_path is None:
        # I want to grab the previous folder
        output_folder_path = os.path.dirname(input_folder_path)

    # Start Time
    start_time = time.time()
    print("Finding images...")
    image_files = sorted(glob.glob(f'{input_folder_path}/*.png'))
    if not image_files:
        print(f"No images found in {input_folder_path}")
        return False

    n_total_images = len(image_files)
    print(f"Found {n_total_images} images in {input_folder_path}")

    images_per_page = rows * columns
    n_pages = math.ceil(n_total_images / images_per_page)
    print(f"Creating {n_pages} pages with {rows}x{columns} grid layout")

    for page in tqdm(range(n_pages), desc="Creating gallery pages", unit="page"):
        start_idx = page * images_per_page
        end_idx = min(start_idx + images_per_page, n_total_images)
        page_images = end_idx - start_idx

        fig = initialize_page(dpi=dpi)

        for i in range(page_images):
            
            # Get the image file path
            img_idx = start_idx + i
            file_path = image_files[img_idx]
            
            # Create subplot at the calculated position
            r, c = divmod(i, columns)
            subplot_idx = r * columns + c + 1
            ax = fig.add_subplot(rows, columns, subplot_idx)
            basename = os.path.basename(file_path).replace('.png', '')
            img = plt.imread(file_path)
            ax.imshow(img)
            if len(basename) > 25: basename=basename[:25]
            ax.set_title(basename, fontsize=10, pad=2, loc='center')
            ax.axis('off')

        # Turn off any unused subplots
        turn_off_unused_subplots(
            fig, rows,columns, page_images
        )

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.tight_layout()

        # Save as PNG with page number
        page_filename = f"{output_folder_path}/{output_filename}{page+1}.png"
        plt.savefig(
            page_filename, 
            bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        print(f"Saved {page_filename}")

    processing_time = time.time() - start_time
    print(f"Created {n_pages} gallery PNGs with {n_total_images} images in total.")
    print(f"Total processing time: {processing_time:.2f} seconds")
    return True

def display_masks_on_axis(im, masks, ax, title=None):
    """
    Display a grayscale image with overlaid masks in different colors on a given axis.
    """
    ax.imshow(im, cmap='gray')
    if masks.ndim == 3:
        viz.add_masks(masks, ax)
    else: 
        masks = viz.masks_to_3d_array(masks)
        viz.add_masks(masks, ax)
    ax.axis('off')
    if title is not None:
        ax.set_title(title, fontsize=10, pad=2, loc='center')


def convert_zarr_to_gallery(
    input_zarr_path, 
    output_folder_path: str = None,
    rows=4, columns=4, dpi=150
    ):
    """
    Convert a Zarr Predictions file to a gallery of PNG images.
    """
    # Initialize Output Filename
    output_filename = "gallery_p"
    if output_folder_path is None:
        output_folder_path = input_zarr_path.replace('.zarr', '_gallery')
    os.makedirs(output_folder_path, exist_ok=True)

    # Start Time
    start_time = time.time()
    print("Finding images...")
    if not os.path.exists(input_zarr_path):
        print(f"Input Zarr file not found: {input_zarr_path}")
        return False

    zfile = zarr.open(input_zarr_path, mode='r')
    run_ids = list(zfile.keys())

    n_total_images = len(run_ids)
    print(f"Found {n_total_images} images in {input_zarr_path}")

    images_per_page = rows * columns
    n_pages = math.ceil(n_total_images / images_per_page)
    print(f"Creating {n_pages} pages with {rows}x{columns} grid layout")    

    for page in tqdm(range(n_pages), desc="Converting Zarr to Gallery", unit="page"):

        start_idx = page * images_per_page
        end_idx = min(start_idx + images_per_page, n_total_images)
        page_images = end_idx - start_idx

        fig = initialize_page(dpi=dpi)

        for i in range(page_images):
            run_id = run_ids[start_idx + i]
            img = zfile[run_id]['image'][:]
            masks = zfile[run_id]['masks'][:]

            r, c = divmod(i, columns)
            subplot_idx = r * columns + c + 1
            ax = fig.add_subplot(rows, columns, subplot_idx)
            ax.imshow(img, cmap='gray')
            if masks is not None:
                # If you want to overlay masks, use your show_anns function
                if len(run_id) > 25: run_id=run_id[:25]
                display_masks_on_axis(img, masks, ax, title=run_id)

        # Turn off any unused subplots
        turn_off_unused_subplots(
            fig, rows, columns, page_images
        )
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.tight_layout()

        page_filename = f"{output_folder_path}/{output_filename}{page+1}.png"
        plt.savefig(
            page_filename, 
            bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        print(f"Saved {page_filename}")

    processing_time = time.time() - start_time
    print(f"Created {n_pages} gallery PNGs with {n_total_images} images in total.")
    print(f"Total processing time: {processing_time:.2f} seconds")
    return True        