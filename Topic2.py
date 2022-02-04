from math import sqrt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from csv import reader
from math import sqrt
import random
import pygame
import random
import sys
import math
from array import *
import queue as q


# region SearchAlgorithms
class Node:
    id = None  # Unique value for each node.
    up = None  # Represents value of neighbors (up, down, left, right).
    down = None
    left = None
    right = None
    previousNode = None  # Represents value of neighbors.
    nodeCost = None

    def __init__(self, value):
        self.value = value


class SearchAlgorithms:
    ''' * DON'T change Class, Function or Parameters Names and Order
        * You can add ANY extra functions,
          classes you need as long as the main
          structure is left as is '''
    path = []  # Represents the correct path from start node to the goal node.
    fullPath = []  # Represents all visited nodes from the start node to the goal node.
    maze = ''
    nodeMaze = []
    goal = 'E'

    def __init__(self, mazeStr):
        ''' mazeStr contains the full board
        The board is read row wise,
        the nodes are numbered 0-based starting
        the leftmost node'''
        self.maze = mazeStr
        pass

    rowLength = len(maze.split(' '))

    def MazeBoard(self):
        myMaze = []
        myMaze.clear()
        id = 0
        arr = []
        nodeList = []
        self.nodeMaze.clear()

        for i in range(len(self.maze)):
            # if not any((c in chars) for c in str):
            if self.maze[i] == ',':
                continue
            elif self.maze[i] == ' ':
                myMaze.append(arr.copy())
                arr.clear()
            else:
                arr.append(self.maze[i])
        myMaze.append(arr.copy())
        myMaze = np.array(myMaze)
        # print(myMaze)

        for i in range(len(myMaze)):
            for j in range(len(myMaze[i])):
                myNode = Node(myMaze[i][j])
                myNode.id = id
                myNode.previousNode = myNode.id - 1
                if i - 1 >= 0 and i - 1 < len(myMaze) and j <= len(myMaze) and j >= 0:
                    myNode.up = id - len(myMaze[i])
                if i + 1 < len(myMaze) and i + 1 >= 0 and j <= len(myMaze) and j >= 0:
                    myNode.down = id + len(myMaze[i])
                if j - 1 >= 0 and j <= len(myMaze) and i <= len(myMaze) and i >= 0:
                    myNode.left = id - 1
                if j + 1 < len(myMaze[i]) and i >= 0 and j >= 0 and i <= len(myMaze):
                    myNode.right = id + 1

                nodeList.append(myNode)
                id += 1

            self.nodeMaze.append(nodeList.copy())
            nodeList.clear()
        self.nodeMaze.append(nodeList.copy())
        finalMaze = np.array(self.nodeMaze)
        # print("nodemaze:",finalMaze)

    def UCS(self):

        pQueue = q.PriorityQueue()
        visited = set()
        a, b, arr = pQueue.get()
        arr.clear()
        for i in range(len(self.nodeMaze)):
            for j in range(len(self.nodeMaze[i])):
                if self.nodeMaze[i][j].value == 'S':
                    startNode = self.nodeMaze[i][j]
                    break
        pQueue.put((0, startNode, [startNode]))
        while pQueue:
            cost, myNode, self.path = pQueue.get()
            visited.add(myNode)
            if myNode.value == self.goal:
                return self.path

            self.path.append(myNode.id)
        else:
            for i in range(len(self.nodeMaze)):
                for j in range(len(self.nodeMaze[i])):
                    if self.nodeMaze[i][j].id == myNode.up and myNode.up != None:
                        if self.nodeMaze[i][j] not in visited:
                            if self.nodeMaze[i][j].value == '.':
                                visited.add(self.nodeMaze[i][j])
                                self.nodeMaze[i][j].previousNode = myNode.id
                    if self.nodeMaze[i][j].id == myNode.down:
                        if self.nodeMaze[i][j] not in visited and myNode.down != None:
                            if self.nodeMaze[i][j].value == '.':
                                visited.add(self.nodeMaze[i][j])
                                self.nodeMaze[i][j].previousNode = myNode.id
                    if self.nodeMaze[i][j].id == myNode.left and myNode.left != None:
                        if self.nodeMaze[i][j] not in visited:
                            if self.nodeMaze[i][j].value == '.':
                                visited.add(self.nodeMaze[i][j])
                                self.nodeMaze[i][j].previousNode = myNode.id
                    if self.nodeMaze[i][j].id == myNode.right and myNode.right != None:
                        if self.nodeMaze[i][j] not in visited:
                            visited.add(self.nodeMaze[i][j])
                            if self.nodeMaze[i][j].value == '.':
                                visited.add(self.nodeMaze[i][j])
                                self.nodeMaze[i][j].previousNode = myNode.id
                    self.path.append(myNode.id)

    def DFS(self):
        # Fill the correct path in self.path
        # self.fullPath should contain the order of visited nodes
        # self.path should contain the direct path from start node to goal node

        # self.UCS()
        self.MazeBoard()
        visited = []
        notVisited = []

        for i in range(len(self.nodeMaze)):
            for j in range(len(self.nodeMaze[i])):
                if self.nodeMaze[i][j].value == 'S':
                    myNodee = self.nodeMaze[i][j]
                    break
        notVisited.append(myNodee)

        while notVisited:

            myNodee = notVisited.pop()
            visited.append(myNodee)

            self.fullPath.append(myNodee.id)

            if myNodee.previousNode != -1:
                self.path.append(myNodee.previousNode)

            if myNodee.value == self.goal:
                self.fullPath.append(myNodee.id)
                break

            else:
                for i in range(len(self.nodeMaze)):
                    for j in range(len(self.nodeMaze[i])):
                        if self.nodeMaze[i][j].id == myNodee.up and myNodee.up != None:
                            if self.nodeMaze[i][j] not in visited:
                                if self.nodeMaze[i][j].value != '#':
                                    notVisited.append(self.nodeMaze[i][j])
                                    visited.append(self.nodeMaze[i][j])
                                if self.nodeMaze[i][j].value == 'E':
                                    break
                        if self.nodeMaze[i][j].id == myNodee.down and myNodee.down != None:
                            if self.nodeMaze[i][j] not in visited:
                                if self.nodeMaze[i][j].value != '#':
                                    notVisited.append(self.nodeMaze[i][j])
                                    visited.append(self.nodeMaze[i][j])
                                if self.nodeMaze[i][j].value == 'E':
                                    break
                        if self.nodeMaze[i][j].id == myNodee.left and Node.left != None:
                            if self.nodeMaze[i][j] not in visited:
                                if self.nodeMaze[i][j].value != '#':
                                    notVisited.append(self.nodeMaze[i][j])
                                    visited.append(self.nodeMaze[i][j])
                                if self.nodeMaze[i][j].value == 'E':
                                    break
                        if self.nodeMaze[i][j].id == myNodee.right and myNodee.right != None:
                            if self.nodeMaze[i][j] not in visited:
                                if self.nodeMaze[i][j].value != '#':
                                    notVisited.append(self.nodeMaze[i][j])
                                    visited.append(self.nodeMaze[i][j])
                                    if self.nodeMaze[i][j].value == 'E':
                                        break

                        # notVisited.reverse()
                notVisited.reverse()

        return self.fullPath, self.path


# endregion

# region Gamings
class Gaming:
    def __init__(self):
        self.COLOR_BLUE = (0, 0, 240)
        self.COLOR_BLACK = (0, 0, 0)
        self.COLOR_RED = (255, 0, 0)
        self.COLOR_YELLOW = (255, 255, 0)

        self.Y_COUNT = int(5)
        self.X_COUNT = int(8)

        self.PLAYER = 0
        self.AI = 1

        self.PLAYER_PIECE = 1
        self.AI_PIECE = 2

        self.WINNING_WINDOW_LENGTH = 3
        self.EMPTY = 0
        self.WINNING_POSITION = []
        self.SQUARESIZE = 80

        self.width = self.X_COUNT * self.SQUARESIZE
        self.height = (self.Y_COUNT + 1) * self.SQUARESIZE

        self.size = (self.width, self.height)

        self.RADIUS = int(self.SQUARESIZE / 2 - 5)

        self.screen = pygame.display.set_mode(self.size)

    def create_board(self):
        board = np.zeros((self.Y_COUNT, self.X_COUNT))
        return board

    def drop_piece(self, board, row, col, piece):
        board[row][col] = piece

    def is_valid_location(self, board, col):
        return board[self.Y_COUNT - 1][col] == 0

    def get_next_open_row(self, board, col):
        for r in range(self.Y_COUNT):
            if board[r][col] == 0:
                return r

    def print_board(self, board):
        print(np.flip(board, 0))

    def winning_move(self, board, piece):
        self.WINNING_POSITION.clear()
        for c in range(self.X_COUNT - 2):
            for r in range(self.Y_COUNT):
                if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece:
                    self.WINNING_POSITION.append([r, c])
                    self.WINNING_POSITION.append([r, c + 1])
                    self.WINNING_POSITION.append([r, c + 2])
                    return True

        for c in range(self.X_COUNT):
            for r in range(self.Y_COUNT - 2):
                if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece:
                    self.WINNING_POSITION.append([r, c])
                    self.WINNING_POSITION.append([r + 1, c])
                    self.WINNING_POSITION.append([r + 2, c])
                    return True

        for c in range(self.X_COUNT - 2):
            for r in range(self.Y_COUNT - 2):
                if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece:
                    self.WINNING_POSITION.append([r, c])
                    self.WINNING_POSITION.append([r + 1, c + 1])
                    self.WINNING_POSITION.append([r + 2, c + 2])
                    return True

        for c in range(self.X_COUNT - 2):
            for r in range(2, self.Y_COUNT):
                if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece:
                    self.WINNING_POSITION.append([r, c])
                    self.WINNING_POSITION.append([r - 1, c + 1])
                    self.WINNING_POSITION.append([r - 2, c + 2])
                    return True

    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = self.PLAYER_PIECE
        if piece == self.PLAYER_PIECE:
            opp_piece = self.AI_PIECE

        if window.count(piece) == 3:
            score += 100
        elif window.count(piece) == 2 and window.count(self.EMPTY) == 1:
            score += 5

        if window.count(opp_piece) == 3 and window.count(self.EMPTY) == 1:
            score -= 4

        return score

    def score_position(self, board, piece):
        score = 0

        center_array = [int(i) for i in list(board[:, self.X_COUNT // 2])]
        center_count = center_array.count(piece)
        score += center_count * 3

        for r in range(self.Y_COUNT):
            row_array = [int(i) for i in list(board[r, :])]
            for c in range(self.X_COUNT - 3):
                window = row_array[c: c + self.WINNING_WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        for c in range(self.X_COUNT):
            col_array = [int(i) for i in list(board[:, c])]
            for r in range(self.Y_COUNT - 3):
                window = col_array[r: r + self.WINNING_WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        for r in range(self.Y_COUNT - 3):
            for c in range(self.X_COUNT - 3):
                window = [board[r + i][c + i] for i in range(self.WINNING_WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        for r in range(self.Y_COUNT - 3):
            for c in range(self.X_COUNT - 3):
                window = [board[r + 3 - i][c + i] for i in range(self.WINNING_WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        return score

    def is_terminal_node(self, board):
        return self.winning_move(board, self.PLAYER_PIECE) or self.winning_move(board, self.AI_PIECE) or len(
            self.get_valid_locations(board)) == 0

    def AlphaBeta(self, board, depth, alpha, beta, currentPlayer):
        valid_locations = self.get_valid_locations(board)
        column = random.choice(valid_locations)

        '''Implement here'''

        if depth == 0:
            return None, self.score_position(board, self.AI_PIECE)

        if self.is_terminal_node(board):
            return None, self.score_position(board, self.AI_PIECE)

        locations = self.get_valid_locations(board)

        if currentPlayer == self.AI:

            finalValue = -math.inf

            for location in locations:
                row = self.get_next_open_row(board, location)
                self.drop_piece(board, row, location, self.AI_PIECE)
                tempCol, value = self.AlphaBeta(board, depth - 1, alpha, beta, self.PLAYER)
                self.drop_piece(board, row, location, 0)

                if value > finalValue:
                    finalValue = value
                    column = location

                alpha = max(alpha, finalValue)
                if beta <= alpha:
                    break
            return column, finalValue

        else:

            finalValue = math.inf

            for location in locations:
                row = self.get_next_open_row(board, location)
                self.drop_piece(board, row, location, self.PLAYER_PIECE)
                tempCol, value = self.AlphaBeta(board, depth - 1, alpha, beta, self.AI)
                self.drop_piece(board, row, location, 0)

                if (value < finalValue):
                    finalValue = value
                    column = location

                beta = min(beta, finalValue)
                if beta <= alpha:
                    break

        return column, finalValue

    def get_valid_locations(self, board):
        valid_locations = []
        for col in range(self.X_COUNT):
            if self.is_valid_location(board, col):
                valid_locations.append(col)
        return valid_locations

    def pick_best_move(self, board, piece):
        best_score = -10000
        valid_locations = self.get_valid_locations(board)
        best_col = random.choice(valid_locations)

        for col in valid_locations:
            row = self.get_next_open_row(board, col)
            temp_board = board.copy()
            self.drop_piece(temp_board, row, col, piece)
            score = self.score_position(temp_board, piece)

            if score > best_score:
                best_score = score
                best_col = col

        return best_col

    def draw_board(self, board):
        for c in range(self.X_COUNT):
            for r in range(self.Y_COUNT):
                pygame.draw.rect(self.screen, self.COLOR_BLUE,
                                 (c * self.SQUARESIZE, r * self.SQUARESIZE + self.SQUARESIZE, self.SQUARESIZE,
                                  self.SQUARESIZE))
                pygame.draw.circle(self.screen, self.COLOR_BLACK, (
                    int(c * self.SQUARESIZE + self.SQUARESIZE / 2),
                    int(r * self.SQUARESIZE + self.SQUARESIZE + self.SQUARESIZE / 2)),
                                   self.RADIUS)

        for c in range(self.X_COUNT):
            for r in range(self.Y_COUNT):
                if board[r][c] == self.PLAYER_PIECE:
                    pygame.draw.circle(self.screen, self.COLOR_RED, (
                        int(c * self.SQUARESIZE + self.SQUARESIZE / 2),
                        self.height - int(r * self.SQUARESIZE + self.SQUARESIZE / 2)),
                                       self.RADIUS)
                elif board[r][c] == self.AI_PIECE:
                    pygame.draw.circle(self.screen, self.COLOR_YELLOW, (
                        int(c * self.SQUARESIZE + self.SQUARESIZE / 2),
                        self.height - int(r * self.SQUARESIZE + self.SQUARESIZE / 2)),
                                       self.RADIUS)
        pygame.display.update()


# endregion

# region KMEANS
class DataItem:
    def __init__(self, item):
        self.features = item
        self.clusterId = -1

    def getDataset():
        data = []
        data.append(DataItem([0, 0, 0, 0]))
        data.append(DataItem([0, 0, 0, 1]))
        data.append(DataItem([0, 0, 1, 0]))
        data.append(DataItem([0, 0, 1, 1]))
        data.append(DataItem([0, 1, 0, 0]))
        data.append(DataItem([0, 1, 0, 1]))
        data.append(DataItem([0, 1, 1, 0]))
        data.append(DataItem([0, 1, 1, 1]))
        data.append(DataItem([1, 0, 0, 0]))
        data.append(DataItem([1, 0, 0, 1]))
        data.append(DataItem([1, 0, 1, 0]))
        data.append(DataItem([1, 0, 1, 1]))
        data.append(DataItem([1, 1, 0, 0]))
        data.append(DataItem([1, 1, 0, 1]))
        data.append(DataItem([1, 1, 1, 0]))
        data.append(DataItem([1, 1, 1, 1]))
        return data


class Cluster:
    def __init__(self, id, centroid):
        self.centroid = centroid
        self.data = []
        self.id = id

    def update(self, clusterData):
        self.data = []
        for item in clusterData:
            self.data.append(item.features)
        tmpC = np.average(self.data, axis=0)
        tmpL = []
        for i in tmpC:
            tmpL.append(i)
        self.centroid = tmpL


class SimilarityDistance:
    def euclidean_distance(self, p1, p2):
        Result = 0
        for i in range(len(p1)):
            Result += pow(p1[i] - p2[i], 2)
        Final = sqrt(Result)
        return Final
        pass

    def Manhattan_distance(self, p1, p2):
        Result = 0
        for x in range(len(p1)):
            Result += abs(p1[x] - p2[x])
        return Result
        pass


class Clustering_kmeans:
    def __init__(self, data, k, noOfIterations, isEuclidean):
        self.data = data
        self.k = k
        self.distance = SimilarityDistance()
        self.noOfIterations = noOfIterations
        self.isEuclidean = isEuclidean

    def initClusters(self):
        self.clusters = []
        for i in range(self.k):
            self.clusters.append(Cluster(i, self.data[i * 10].features))

    def getClusters(self):
        self.initClusters()
        '''Implement Here'''
        for i in range(self.noOfIterations):
            for j in self.data:
                minimumDistance = math.inf
                for c in self.clusters:
                    euclideanDistance = self.distance.euclidean_distance(c.centroid, j.features)
                    manhattanDitance = self.distance.Manhattan_distance(c.centroid, j.features)
                    if self.isEuclidean:
                        if euclideanDistance < minimumDistance:
                            j.clusterId = c.id
                            minimumDistance = euclideanDistance
                    else:
                        if manhattanDitance < minimumDistance:
                            j.clusterId = c.id
                            minimumDistance = manhattanDitance
                data = [x for x in self.data if x.clusterId == j.clusterId]
                self.clusters[j.clusterId].update(data)
        return self.clusters
        pass


# endregion

#################################### Algorithms Main Functions #####################################
# region Search_Algorithms_Main_Fn
def SearchAlgorithm_Main():
    searchAlgo = SearchAlgorithms('S,.,.,#,.,.,. .,#,.,.,.,#,. .,#,.,.,.,.,. .,.,#,#,.,.,. #,.,#,E,.,#,.')
    fullPath, path = searchAlgo.DFS()
    print('**DFS**\n Full Path is: ' + str(fullPath) + '\n Path is: ' + str(path))


# endregion

# region Gaming_Main_fn
def Gaming_Main():
    game = Gaming()
    board = game.create_board()
    game.print_board(board)
    game_over = False

    pygame.init()

    game.draw_board(board)
    pygame.display.update()

    myfont = pygame.font.SysFont("monospace", 50)

    turn = 1

    while not game_over:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.MOUSEMOTION:
                pygame.draw.rect(game.screen, game.COLOR_BLACK, (0, 0, game.width, game.SQUARESIZE))
                posx = event.pos[0]
                if turn == game.PLAYER:
                    pygame.draw.circle(game.screen, game.COLOR_RED, (posx, int(game.SQUARESIZE / 2)), game.RADIUS)

            pygame.display.update()

            if event.type == pygame.MOUSEBUTTONDOWN:
                pygame.draw.rect(game.screen, game.COLOR_BLACK, (0, 0, game.width, game.SQUARESIZE))

                if turn == game.PLAYER:
                    posx = event.pos[0]
                    col = int(math.floor(posx / game.SQUARESIZE))

                    if game.is_valid_location(board, col):
                        row = game.get_next_open_row(board, col)
                        game.drop_piece(board, row, col, game.PLAYER_PIECE)

                        if game.winning_move(board, game.PLAYER_PIECE):
                            label = myfont.render("Player Human wins!", 1, game.COLOR_RED)
                            print(game.WINNING_POSITION)
                            game.screen.blit(label, (40, 10))
                            game_over = True

                        turn += 1
                        turn = turn % 2

                        # game.print_board(board)
                        game.draw_board(board)

        if turn == game.AI and not game_over:

            col, minimax_score = game.AlphaBeta(board, 5, -math.inf, math.inf, True)

            if game.is_valid_location(board, col):
                row = game.get_next_open_row(board, col)
                game.drop_piece(board, row, col, game.AI_PIECE)

                if game.winning_move(board, game.AI_PIECE):
                    label = myfont.render("Player AI wins!", 1, game.COLOR_YELLOW)
                    print(game.WINNING_POSITION)
                    game.screen.blit(label, (40, 10))
                    game_over = True

                # game.print_board(board)
                game.draw_board(board)

                turn += 1
                turn = turn % 2

        if game_over:
            pygame.time.wait(3000)
            return game.WINNING_POSITION


# endregion


# region KMeans_Main_Fn
def Kmeans_Main():
    dataset = DataItem.getDataset()
    # 1 for Euclidean and 0 for Manhattan
    clustering = Clustering_kmeans(dataset, 2, len(dataset), 1)
    clusters = clustering.getClusters()
    for cluster in clusters:
        for i in range(4):
            cluster.centroid[i] = round(cluster.centroid[i], 2)
        print(cluster.centroid[:4])
    return clusters


# endregion


######################## MAIN ###########################33
if __name__ == '__main__':
    SearchAlgorithm_Main()
    Gaming_Main()
    Kmeans_Main()

