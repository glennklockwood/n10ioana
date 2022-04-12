"""Common utilizies used across analyses.
"""

def humanize_bytes(quantity):
    """Converts a quantity of bytes into a human-readable format.

    Args:
        quantity (int): Quantity of bytes

    Returns:
        tuple of (float, str): the number and units of a more human-readable
            representation of quantity.  The first element will always be
            a float between 1.0 and 1024.0.
    """
    return_unit = ''
    if quantity % 1024 != 0:
        return quantity, return_unit

    for unit in ['K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if quantity >= 1024.0:
            quantity /= 1024.0
            return_unit = unit + 'iB'
        else:
            break

    return quantity, return_unit
