from saber import cli_context
import click

@click.command(context_settings=cli_context)
@click.option("-i", "--input", type=str, required=True, 
              help="Path to the Zarr file.")
def class_info(input):
    """
    Print information about the classes in a readable format.
    """
    print_class_info(input)

def print_class_info(input):
    import zarr, json

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
