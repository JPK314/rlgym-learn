# pyright: reportUnusedParameter=false

from typing import Generic, TypeVar

T = TypeVar("T")


class PythonSerde(Generic[T]):
    def append(self, buf: memoryview, offset: int, obj: T) -> int:
        """
        Appends bytes of obj to buf starting at offset.
        :param buf: a memoryview to write into (DO NOT hold a reference to this memory view after this function ends!)
        :param offset: an offset into the memory view to start writing
        :param obj: the obj to write as bytes
        :return: new offset after appending bytes
        """
        raise NotImplementedError

    def get_bytes(self, start_addr: int | None, obj: T) -> bytes:
        """
        :param start_addr: the starting address for where the returned bytes will be written. May be None in contexts where there is no guaranteed start address.
        :param obj: the obj to write as bytes
        :return: bytes for obj
        """
        raise NotImplementedError

    def retrieve(self, buf: memoryview, offset: int) -> tuple[T, int]:
        """
        Retrieves obj encoded using self.append or self.get_bytes from the buffer starting at offset.
        :param buf: a memoryview to read from (DO NOT hold a reference to this memory view after this function ends!)
        :param offset: an offset into the memory view to start reading
        :return: Tuple of obj and the offset into the memory view after retrieving obj
        """
        raise NotImplementedError
