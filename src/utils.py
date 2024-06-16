

def get_path(id: str, train: bool = True) -> str:
    """
    Get the path of the Gravitational Wave data.

    :param id:
    :param train:
    :return:
    """
    which: str = 'train' if train else 'test'
    return f'/{which}/{id[0]}/{id[1]}/{id[2]}/{id}.npy'