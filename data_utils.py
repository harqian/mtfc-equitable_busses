import pathlib

def find_data_file(name: str) -> pathlib.Path:
    candidates = [pathlib.Path('/data') / name, pathlib.Path('data') / name, pathlib.Path('/sfmta_data/') / name, pathlib.Path('sfmta_data/') / name, pathlib.Path(name), pathlib.Path(f'/{name}')]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f'Could not find {name} in /data or ./data')