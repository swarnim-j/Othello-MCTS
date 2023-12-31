class Board:
    DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (-1, 1), (1, -1)]

    def __init__(self, size):
        self.n = size
        
        self.pieces = [[0] * size for _ in range(size)]

        self.board[size//2 - 1][size//2 - 1] = 1
        self.board[size//2][size//2] = 1
        self.board[size//2 - 1][size//2] = -1
        self.board[size//2][size//2 - 1] = -1

    def getBoardSize(self):
        return self.n
    
    def diff(self, colour):
        return sum([row.count(colour) - row.count(-colour) for row in self.pieces])
    
    def isValidMove(self, x, y, dx, dy, colour):
        if x + dx >= self.n or x + dx < 0 or y + dy >= self.n or y + dy < 0:
            return False
        if self.pieces[x + dx][y + dy] == colour:
            return False
        for i in range(2, self.n):
            if self.pieces[x + i * dx][y + i * dy] == 0:
                return False
            if self.pieces[x + i * dx][y + i * dy] == colour:
                return True
        return False

    def getValidMoves(self, colour):
        moves = []
        for x in range(self.n):
            for y in range(self.n):
                if self.pieces[x][y] != 0:
                    continue
                for dx, dy in self.DIRECTIONS:
                    if self.isValidMove(x, y, dx, dy, colour):
                        moves.append((x, y))
                        break
        return moves