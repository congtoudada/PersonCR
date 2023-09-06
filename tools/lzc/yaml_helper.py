import yaml

def read_yaml(file, encoding='utf-8'):
    with open(file, encoding=encoding) as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)

def write_yaml(file, wtdata, encoding='utf-8'):
    with open(file, encoding=encoding, mode='w') as f:
        yaml.dump(wtdata, stream=f, allow_unicode=True)