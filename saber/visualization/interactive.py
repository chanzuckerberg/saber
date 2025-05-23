from IPython.display import display, Javascript
from saber.visualization import embeddings
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import ipywidgets as widgets
import numpy as np
import torch

# Function to handle keyboard events (left/right arrow keys)
def on_key_event(event):
    if event['event'] == 'keydown':
        if event['name'] == 'ArrowLeft' and slider.value > slider.min:
            slider.value -= 1  # Move slider left
        elif event['name'] == 'ArrowRight' and slider.value < slider.max:
            slider.value += 1  # Move slider right

# Register keyboard events via JavaScript (works in Colab/Jupyter Notebook)
def register_keyboard_events():
    display(Javascript("""
        window.addEventListener('keydown', (event) => {
            if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
                google.colab.kernel.invokeFunction('notebook.on_key_event', [{event: event.type, name: event.key}], {});
            }
        });
    """))

def display_embedding_channel(image, embed, index):
    """
    Display two plots side-by-side:
    - Left: a base image using the first channel
    - Right: the embedding visualization for the selected channel
    Assumes embed has shape [channels, height, width]
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Base image: use channel 0
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Base Image")
    axes[0].axis('off')
    
    # Embedding visualization for the selected channel
    channel_image = embed[index, :, :]
    axes[1].imshow(channel_image, cmap='viridis')
    axes[1].set_title(f"Embedding Channel {index}")
    axes[1].axis('off')
    
def return_channel_slider(method):
    if method == 'embed':
        return widgets.IntSlider(min=0, max=256, step=1, value=128, description="Channel")
    elif method == 'high_res1':
        return widgets.IntSlider(min=0, max=32, step=1, value=16, description="Channel")
    elif method == 'high_res2':
        return widgets.IntSlider(min=0, max=64, step=1, value=32, description="Channel")
    else:
        raise ValueError(f"Invalid method: {method} - must be one of 'embed', 'high_res1', 'high_res2'")
    
def display_sam2_composite_embedding(zfile, run_ids, model, index):
    
    # Get Image and Embedding
    image = zfile[run_ids[index]]['image'][:]
    image = np.repeat(image[..., None], 3, axis=2)
    model.backbone.set_image(image)
    high_res_feats = model.backbone._features["high_res_feats"]
    features = model.backbone._features["image_embed"][0].cpu().numpy()
    # features = high_res_feats[0].cpu().numpy()[0,]
    # features = high_res_feats[1].cpu().numpy()[0,]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Base image: use channel 0
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title(f"Base Image (Run: {run_ids[index]})")
    axes[0].axis('off')
    
    # Embedding visualization for the selected channel
    composite = embeddings.visualize_patch_features(features)
    axes[1].imshow(composite, cmap='viridis')
    axes[1].set_title(f"Embedding Composite")
    axes[1].axis('off')
    
def display_dinov2_composite_embedding(zfile, run_ids, model, index):
    
    # Get Image and Embedding
    image0 = torch.Tensor(zfile[run_ids[index]]['image'][:])
    image = image0.unsqueeze(0).unsqueeze(0)

    with torch.inference_mode():
        image_batch = image.to(torch.float).cuda()
        tokens = model.cuda().get_intermediate_layers(image_batch, norm=True)[0].cpu()
    # List Tokens Shape: [nImages, (nEmbedx, nEmbedy), nChannels]
    # nEmbedx = tokens.shape[1]^{-1/2}
    # nEmbedy = tokens.shape[1]^{-1/2}
    list_tokens = tokens.numpy()
    features = list_tokens[0].reshape((int(list_tokens.shape[1]**0.5), int(list_tokens.shape[1]**0.5), -1))
    features = torch.from_numpy(features).permute(2, 0, 1) # (C, H, W) = [768, 79, 79]
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Base image: use channel 0
    axes[0].imshow(image0, cmap='gray')
    axes[0].set_title(f"Base Image (Run: {run_ids[index]})")
    axes[0].axis('off')
    
    # Embedding visualization for the selected channel
    composite = embeddings.visualize_patch_features(features)
    axes[1].imshow(composite, cmap='viridis')
    axes[1].set_title(f"Embedding Composite")
    axes[1].axis('off')
    
    plt.show()
    
# Define unique colors for each class (RGBA values)
colors = [
    (1, 0, 0, 0.5),  # Red with transparency    
    (0, 1, 0, 0.5),  # Green with transparency    
    (0, 0, 1, 0.5),  # Blue with transparency
    (1, 1, 0, 0.5),  # Yellow with transparency
]

def show_dataset(dataset, index):
    dict = dataset.__getitem__(index)
    im = dict['image'][0, ...]
    mask = dict['mask'][0, ...]
    value = dict['label'].item()

    # Create a custom colormap for this mask
    custom_cmap = ListedColormap([
        (1, 1, 1, 0),  # Transparent white for 0 values
        colors[value]  # Assigned color for non-zero values
    ])

    if mask.max() == 0:
        print('BAD MASK!!')
    else:
        plt.figure(figsize=(8,8))
        plt.imshow(im, cmap='gray'); plt.axis('off')

        # Overlay the mask with the specific color
        plt.imshow(mask, cmap=custom_cmap, alpha=0.6)