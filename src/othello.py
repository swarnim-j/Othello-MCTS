class Board:
    DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    def __init__(self, n):
        self.n = n
        self.pieces = [[0] for _ in range(n)]
        mid = self.n // 2
        self.pieces[mid - 1][mid] = 1
        self.pieces[mid][mid - 1] = 1
        self.pieces[mid - 1][mid - 1] = -1
        self.pieces[mid][mid] = -1

    def __getitem__(self, index):
        return self.pieces[index]
    
    def has_legal_moves(self, player_color):
        for row in range(self.n):
            for col in range(self.n):
                if self[row][col] == player_color and self.get_moves((row, col)):
                    return True
        return False
    
    def get_legal_moves(self, position):
        x, y = position
        color = self[x][y]   
        if color == 0:
            return []
        legal_moves = []
        for direction in Board.DIRECTIONS:
            move = self.find_valid_move(position, direction)
            if move:
                x, y = move
                legal_moves.append(x * self.n + y)
        return legal_moves
    
    def find_valid_move(self, origin, direction):
        x, y = origin
        color = self[x][y]
        flips = []
        for x, y in self.increment_move(origin, direction):
            if self[x][y] == 0:
                return (x, y) if flips else None
            elif self[x][y] == color:
                return None
            elif self[x][y] == -color:
                flips.append((x, y))

    def get_flips(self, origin, direction, color):
        flips = []
        for x, y in self.increment_move(origin, direction):
            if self[x][y] == 0:
                return []
            if self[x][y] == -color:
                flips.append((x, y))
            elif self[x][y] == color and flips:
                return flips
        return []
    
    def increment_move(self, origin, direction):
        x, y = origin
        dx, dy = direction
        x += dx
        y += dy
        while 0 <= x < self.n and 0 <= y < self.n:
            yield x, y
            x += dx
            y += dy
            
    def move(self, position, color):
        x, y = position
        self[x][y] = color
        for direction in Board.DIRECTIONS:
            flips = self.get_flips(position, direction, color)
            for x, y in flips:
                self[x][y] = color

    def count(self, color):
        return sum(row.count(color) for row in self.pieces)