import pytest
from yallm.memory.memory_pool import ReqToTokenPool

class TestReqToTokenPool:
    def test_alloc_free(self):
        pool = ReqToTokenPool(10, 10, "cpu")
        assert pool.alloc(1) == [0]
        assert pool.free_size == 9
        assert pool.alloc(1) == [1]
        assert pool.free_size == 8
        assert pool.alloc(1) == [2]
