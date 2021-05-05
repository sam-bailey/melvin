from typing import Any, Callable


class ObjectTagger:
    def __init__(self):
        self._distributions = {}

    def __call__(self, dist_name: str) -> Callable[[Any], Any]:
        def _decorator(cls: Any) -> Any:
            self._distributions[dist_name] = cls
            return cls

        return _decorator

    def get(self, dist_name: str) -> Any:
        return self._distributions[dist_name]
