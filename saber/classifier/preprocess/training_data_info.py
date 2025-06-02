import click, zarr, json


@click.command(context_settings={"show_default": True})
@click.option("--input", type=str, required=True, 
              help="Path to the Zarr file.")
def class_info(input):
    """
    Print information about the classes in a readable format.
    """

    # Load the Zarr file
    zarr_root = zarr.open(input, mode='r')
    try:
        class_dict = json.loads(zarr_root.attrs['class_names'])
    except:
        class_dict = json.loads(zarr_root.attrs['class_dict'])

    print("\nClass Information:")
    print("-" * 50)
    for class_name, class_data in class_dict.items():
        print(f"\nValue: {class_data['value']} - Class: {class_name}")
        print("-" * 50)
