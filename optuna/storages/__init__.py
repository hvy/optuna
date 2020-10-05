from typing import Union

from optuna.core.storages._base import BaseStorage
from optuna.core.storages._in_memory import InMemoryStorage
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._rdb.storage import RDBStorage
from optuna.storages._redis import RedisStorage


# TODO(hvy): Should be possible to register storage string to class mappings to core.
# Then, `get_storage` could be moved to core.
def get_storage(storage: Union[None, str, BaseStorage]) -> BaseStorage:

    if storage is None:
        return InMemoryStorage()
    if isinstance(storage, str):
        if storage.startswith("redis"):
            return RedisStorage(storage)
        else:
            return _CachedStorage(RDBStorage(storage))
    elif isinstance(storage, RDBStorage):
        return _CachedStorage(storage)
    else:
        return storage
