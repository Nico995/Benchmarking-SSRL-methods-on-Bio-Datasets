available_methods = ['rotation']
available_datasets = ['kather']


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

