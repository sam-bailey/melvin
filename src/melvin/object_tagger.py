from typing import Any, Callable


class ObjectTagger:
    def __init__(self):
        self._objects = dict()

    def __call__(self, object_name: str) -> Callable[[Any], Any]:
        def _decorator(obj: Any) -> Any:
            self._objects[object_name] = obj
            return obj

        return _decorator

    def get(self, object_name: str) -> Any:
        return self._objects[object_name]
