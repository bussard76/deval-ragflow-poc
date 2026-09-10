class DevalError(Exception):
    """Base class for expected application errors."""


class ConfigurationError(DevalError):
    pass


class DependencyError(DevalError):
    pass


class ExtractionError(DevalError):
    pass


class RegistryError(DevalError):
    pass


class AdapterError(DevalError):
    pass


class RAGFlowHTTPError(AdapterError):
    def __init__(self, status_code, message, body=None):
        self.status_code = status_code
        self.body = body
        super().__init__("RAGFlow HTTP {}: {}".format(status_code, message))


class RAGFlowBusinessError(AdapterError):
    def __init__(self, code, message, raw=None):
        self.code = code
        self.message = message
        self.raw = raw
        super().__init__("RAGFlow business error {}: {}".format(code, message))


class DatasetConfigurationError(AdapterError):
    pass


class ReconciliationError(AdapterError):
    pass


class UnsafeOperation(DevalError):
    pass


class UnknownRemoteState(AdapterError):
    pass
