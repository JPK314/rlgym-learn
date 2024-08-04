from __future__ import annotations

from abc import abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class TypeSerde(Generic[T]):
    @abstractmethod
    def to_bytes(self, obj: T) -> bytes:
        """
        Function to convert obj to bytes, for passing between batched agent and the agent manager.
        :return: bytes b such that from_bytes(b) == obj.
        """
        raise NotImplementedError

    @abstractmethod
    def from_bytes(self, byts: bytes) -> T:
        """
        Function to convert bytes to T, for passing between batched agent and the agent manager.
        :return: T obj such that from_bytes(to_bytes(obj)) == obj.
        """
        raise NotImplementedError
