import pytest

from rmd.parallel.sharding import round_robin_indices


def test_round_robin_indices_cover_prompts_once():
    shards = [round_robin_indices(10, rank, 3) for rank in range(3)]

    assert shards == [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]]
    assert sorted(index for shard in shards for index in shard) == list(range(10))


def test_round_robin_indices_validates_rank():
    with pytest.raises(ValueError):
        round_robin_indices(10, 3, 3)
