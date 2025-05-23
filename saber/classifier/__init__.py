# Callback to ensure the --multiple-slabs parameter is an odd integer (and at least 1)
def validate_odd(ctx, param, value):
    if value < 0:
        raise click.BadParameter("The --multiple-slabs parameter must be at least 1.")
    if value % 2 == 0:
        raise click.BadParameter("The --multiple-slabs parameter must be an odd number.")
    return value