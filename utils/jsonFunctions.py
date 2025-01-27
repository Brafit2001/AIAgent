import json


def write_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def open_json(file_path) -> dict:
    with open(file_path, "r") as f:
        datos_json = json.load(f)
    return datos_json

