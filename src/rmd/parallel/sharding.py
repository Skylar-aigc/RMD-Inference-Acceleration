"""Small dependency-free helpers shared by distributed CLI paths."""


def round_robin_indices(total: int, rank: int, world_size: int) -> list[int]:
    """Return stable global indices assigned to one data-parallel rank."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return list(range(rank, total, world_size))
