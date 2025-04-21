from src.game.snake_game import Point, Direction


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
        # Peligro: frente
        (dir_r and game._is_collision(Point(head.x + 1, head.y)))
        or (dir_l and game._is_collision(Point(head.x - 1, head.y)))
        or (dir_u and game._is_collision(Point(head.x, head.y - 1)))
        or (dir_d and game._is_collision(Point(head.x, head.y + 1))),
        # Peligro: derecha
        (dir_u and game._is_collision(Point(head.x + 1, head.y)))
        or (dir_d and game._is_collision(Point(head.x - 1, head.y)))
        or (dir_l and game._is_collision(Point(head.x, head.y - 1)))
        or (dir_r and game._is_collision(Point(head.x, head.y + 1))),
        # Peligro: izquierda
        (dir_d and game._is_collision(Point(head.x + 1, head.y)))
        or (dir_u and game._is_collision(Point(head.x - 1, head.y)))
        or (dir_r and game._is_collision(Point(head.x, head.y - 1)))
        or (dir_l and game._is_collision(Point(head.x, head.y + 1))),
        # Dirección actual
        dir_r,
        dir_d,
        dir_l,
        dir_u,
        # Comida
        game.food.x < game.head.x,  # izquierda
        game.food.x > game.head.x,  # derecha
        game.food.y < game.head.y,  # arriba
        game.food.y > game.head.y,  # abajo
    ]

    return list(map(int, state))
