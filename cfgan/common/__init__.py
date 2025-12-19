import inspect
from typing import Callable, List, Any


def restrict_values(param_name: str, valid_values: List[Any]) -> Callable:

    def decorator(func: Callable) -> Callable:
        signature = inspect.signature(func)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # get the parameter value
            if param_name in kwargs:
                value = kwargs[param_name]
            else:
                # get the value from positional arguments
                try:
                    param_index = list(signature.parameters.keys()).index(param_name)
                    if param_index < len(args):
                        value = args[param_index]
                    else:
                        param = signature.parameters[param_name]
                        if param.default is not inspect.Parameter.empty:
                            value = param.default
                        else:
                            raise ValueError(f"Parameter {param_name} is required")
                except (KeyError, ValueError):
                    try:
                        param_index = func.__code__.co_varnames.index(param_name)
                        if param_index < len(args):
                            value = args[param_index]
                        else:
                            raise ValueError(f"Cannot determine value for parameter {param_name}")
                    except (ValueError, IndexError):
                        raise ValueError(f"Cannot determine value for parameter {param_name}")
            # check if the parameter value is within the valid range
            if value not in valid_values:
                raise ValueError('Invalid value for {}. Choose from {}'.format(param_name, valid_values))
            return func(*args, **kwargs)

        return wrapper

    return decorator
