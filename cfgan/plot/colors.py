BLACK = "k"
SILVER = "silver"
WHITE = "w"
BLUE = "#1A94BC"
RED = "#ED556A"
GREEN = "#2BAE85"
YELLOW = "#E8B004"

COLOR_LIST = [BLUE, RED, GREEN, YELLOW]


def get_color(idx: int) -> str:
    return COLOR_LIST[idx % len(COLOR_LIST)]
