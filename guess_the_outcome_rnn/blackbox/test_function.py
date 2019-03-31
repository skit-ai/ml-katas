def box(str1):
    """
    Function returns 0 or 1

    if str1 is longer than 15 chars return 1
    else 0

    :param str1: str
    :return: int
    """
    return int(len(str1) > 15)
