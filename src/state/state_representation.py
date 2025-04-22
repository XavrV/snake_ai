from src.game.snake_game import Point, Direction
import numpy as np


def get_state(game):
    head = game.snake[0]
    point_l = Point(head.x - 1, head.y)
    point_r = Point(head.x + 1, head.y)
    point_u = Point(head.x, head.y - 1)
    point_d = Point(head.x, head.y + 1)

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        (dir_r and game._is_collision(Point(head.x + 1, head.y)))
        or (dir_l and game._is_collision(Point(head.x - 1, head.y)))
        or (dir_u and game._is_collision(Point(head.x, head.y - 1)))
        or (dir_d and game._is_collision(Point(head.x, head.y + 1))),
        (dir_u and game._is_collision(Point(head.x + 1, head.y)))
        or (dir_d and game._is_collision(Point(head.x - 1, head.y)))
        or (dir_l and game._is_collision(Point(head.x, head.y - 1)))
        or (dir_r and game._is_collision(Point(head.x, head.y + 1))),
        (dir_d and game._is_collision(Point(head.x + 1, head.y)))
        or (dir_u and game._is_collision(Point(head.x - 1, head.y)))
        or (dir_r and game._is_collision(Point(head.x, head.y - 1)))
        or (dir_l and game._is_collision(Point(head.x, head.y + 1))),
        dir_r,
        dir_d,
        dir_l,
        dir_u,
        game.food.x < game.head.x,
        game.food.x > game.head.x,
        game.food.y < game.head.y,
        game.food.y > game.head.y,
    ]

    return list(map(int, state))


def get_state2(game):
    head = game.snake[0]

    def danger_at(offset):
        pt = Point(head.x + offset[0], head.y + offset[1])
        return int(game._is_collision(pt))

    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    def rotate_dir(offset, clockwise_steps):
        x, y = offset
        for _ in range(clockwise_steps):
            x, y = y, -x
        return x, y

    def get_direction_vector():
        if dir_r:
            return (1, 0)
        if dir_l:
            return (-1, 0)
        if dir_u:
            return (0, -1)
        if dir_d:
            return (0, 1)

    forward = get_direction_vector()
    right = rotate_dir(forward, 1)
    left = rotate_dir(forward, -1)

    danger_1 = [danger_at(forward), danger_at(right), danger_at(left)]
    danger_2 = [
        danger_at((forward[0] * 2, forward[1] * 2)),
        danger_at((right[0] * 2, right[1] * 2)),
        danger_at((left[0] * 2, left[1] * 2)),
    ]

    dir_state = [int(dir_r), int(dir_d), int(dir_l), int(dir_u)]

    food_left = int(game.food.x < head.x)
    food_right = int(game.food.x > head.x)
    food_up = int(game.food.y < head.y)
    food_down = int(game.food.y > head.y)

    max_dist_x = game.width
    max_dist_y = game.height
    norm_dx = float(abs(game.food.x - head.x)) / max_dist_x
    norm_dy = float(abs(game.food.y - head.y)) / max_dist_y

    food_info = [food_left, food_right, food_up, food_down, norm_dx, norm_dy]

    max_len = game.width * game.height
    length_norm = len(game.snake) / max_len

    def free_directions(offsets):
        return sum(
            1
            for offset in offsets
            if not game._is_collision(Point(head.x + offset[0], head.y + offset[1]))
        )

    free_1 = free_directions([forward, right, left]) / 3
    free_2 = free_directions([(x * 2, y * 2) for (x, y) in [forward, right, left]]) / 3

    state = danger_1 + danger_2 + dir_state + food_info + [length_norm, free_1, free_2]

    assert len(state) == 19
    return np.array(state, dtype=np.float32)


def get_state3(game):
    head = game.snake[0]
    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    def danger(pt):
        return game._is_collision(pt)

    clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
    idx = clock_wise.index(game.direction)

    dir_f = clock_wise[idx]
    dir_r_ = clock_wise[(idx + 1) % 4]
    dir_l_ = clock_wise[(idx - 1) % 4]

    def next_point(direction):
        x, y = head.x, head.y
        if direction == Direction.RIGHT:
            x += 1
        elif direction == Direction.LEFT:
            x -= 1
        elif direction == Direction.UP:
            y -= 1
        elif direction == Direction.DOWN:
            y += 1
        return Point(x, y)

    danger_f = danger(next_point(dir_f))
    danger_r = danger(next_point(dir_r_))
    danger_l = danger(next_point(dir_l_))

    direction_state = [dir_r, dir_d, dir_l, dir_u]

    food = game.food
    food_dir = [food.x < head.x, food.x > head.x, food.y < head.y, food.y > head.y]
    dx = (food.x - head.x) / game.width
    dy = (food.y - head.y) / game.height

    local_map = []
    for dy_ in [-1, 0, 1]:
        for dx_ in [-1, 0, 1]:
            px, py = head.x + dx_, head.y + dy_
            if 0 <= px < game.width and 0 <= py < game.height:
                p = Point(px, py)
                if p == food:
                    local_map.append(2)
                elif p in list(game.snake):
                    local_map.append(1)
                else:
                    local_map.append(0)
            else:
                local_map.append(1)

    state = [
        int(danger_f),
        int(danger_r),
        int(danger_l),
        *map(int, direction_state),
        *map(int, food_dir),
        dx,
        dy,
        *local_map,
    ]
    return np.array(state, dtype=np.float32)
