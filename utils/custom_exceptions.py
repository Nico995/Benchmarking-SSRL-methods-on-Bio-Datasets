available_methods = ['rotation', 'jigsaw', 'autoencoder']
available_datasets = ['pedestrians', 'kather']
available_backbones = ['18', '34']


class MethodNotSupportedError(Exception):
    def __init__(self, method):
        self.message = f"Method '{method}' is not supported. " \
                       f"List of available methods:\n\t{', '.join(available_methods)}"
        super().__init__(self.message)


class DatasetNotSupportedError(Exception):
    def __init__(self, dataset):
        self.message = f"Dataset '{dataset}' is not supported. " \
                       f"List of available datasets:\n\t{', '.join(available_datasets)}"
        super().__init__(self.message)


class BackboneVersionNotSupportedError(Exception):
    def __init__(self, version):
        self.message = f"Resnet '{version}' is not supported. " \
                       f"List of available models:\n\t{', '.join(available_backbones)}"
