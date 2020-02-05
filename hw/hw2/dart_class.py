"""
Author: Pham Nguyen
Last Updated: 02-05-20
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class DartGame:
    """
    Class to play dart game.
    Rules:
        1. A dart is throw randomly at a board. One
           point is given even if the first throw is
           a miss.
        2. If the dart lands in the circle, add one
           point to the score. 
        3. Update the radius of the circle. The
           new radius is determined by a chord that 
           is perpendicular to the line joining the
           center and the position of the dart. The
           chord's midpoint is defined by the dart
           position.
        4. Repeat steps 1-3 until a miss occurs then
           end the game
    """

    def __init__(self, radius=1):
        self.radius = radius
        self.game_result = ()

    @staticmethod
    def new_radius(x, y, r):
        """
        Calculates one-half the length of a chord
        to set as the new radius for the next circle.

        Paramters
        ---------
        x, y: float
            (x, y) coordinates of dart
        r: float
            The radius of the circle

        Returns
        -------
        radius: float
            New radius of the circle
        """
        h = np.sqrt(x**2 + y**2)
        radius = np.sqrt(r**2 - h**2)
        return radius

    def play_round(self, cartesian=True):
        """
        Play a around of darts. A random x and y
        coordinate pair is selected by default. 
        If cartesian is set to FALSE, a random r 
        and angle pair is selected instead.
        """
        win = True
        r = self.radius
        score = 1
        round_list = []
        
        while win is True:
            if cartesian:
                x_rand = np.random.uniform(-self.radius, self.radius)
                y_rand = np.random.uniform(-self.radius, self.radius)
               
            else:
                r_rand = np.random.uniform(0, self.radius)
                theta_rand = np.random.uniform(0, 2*np.pi)
                x_rand = r_rand * np.cos(theta_rand)
                y_rand = r_rand * np.sin(theta_rand)
    
            if x_rand**2 + y_rand**2 <= r**2:
                score += 1
                round_list.append((score, x_rand, y_rand, r))
                r = self.new_radius(x_rand, y_rand, r)
        
            else:
                win = False
                if score > 1:
                    self.game_result = round_list
                else:
                    self.game_result = [(1, x_rand, y_rand, self.radius)]
        return None

    
class DartSimulation:
    
    """
    Class to run a simulation of many dart throwing
    games. Also includes some plotting methods. 
    """
    
    def __init__(self, num_games=1000, radius=1, cartesian=True):
        """
        Class constructor for dart game simulations.
            
        num_games: int
            Total number of games to simulate
        radius: float
            Radius of all board games (does not change!)
        game_list: dict{int: tuple}
            Dictionary containing the results of each
            game. {game #: (score, x position, y position, radius)}
        game_count: int
            Labels games from 1 to num_games
        """
        self.num_games = num_games 
        self.radius = radius
        self.cartesian = cartesian
        self.games_list = {}
        self.game_count = 0
        
    def simulate(self):
        """
        Play N number of games as given by
        the num_games variable.
        """
        game = DartGame(self.radius)
        
        for i in tqdm(range(self.num_games)):
            self.game_count += 1
            game.play_round(self.cartesian)
            self.games_list[self.game_count] = game.game_result
            
        return None
    
    def clear_board(self):
        # Clear all board games from class
        self.games_list.clear()
        return None

    def hist_games(self, save=False):
        """
        Plots a histogram of game final scores. Option
        to save histogram as well.
        """
        scores = [s[-1][0] for s in self.games_list.values()]
            
        high_score = np.max(scores)
        fig = plt.hist(scores, bins=range(high_score), ec="white", alpha=0.9)
        plt.grid(axis="y", color="whitesmoke")
        plt.gca().set_facecolor("whitesmoke")
        plt.gca().tick_params(direction="in", axis="both")
        plt.xlim(0, high_score)
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        print("Mean: {:0.3f}".format(np.mean(scores)))
        print("Std: {:0.3f}".format(np.std(scores)))
        
        if save:
            plt.savefig("histogram_{}.png".format(self.num_games), dpi=150)
            
        plt.show()
        return fig
    
    def scatter_games(self, save=False):
        """
        Make scatter plot of all throws for all games.
        Option to save image.
        """
        x = []
        y = []
        for single_game in self.games_list.values():
            for data in single_game:
                x.append(data[1])
                y.append(data[2])
                
        fig = plt.scatter(x, y, s=10, alpha=0.75)
        plt.tight_layout()
        plt.gca().set_aspect('equal')
        plt.xlim(-self.radius, self.radius)
        plt.ylim(-self.radius, self.radius)
        plt.gca().set_facecolor("whitesmoke")
        plt.gca().tick_params(direction="in", axis="both")
        plt.xlabel("X-position")
        plt.ylabel("Y-position")
        plt.show()
        
        if save:
            plt.savefig("scatter_{}.png".format(self.num_games), dpi=150)
        return fig

    def plot_game(self, number=1, save=False):
        """
        Plot of a single game and save option.
        """
        fig, ax = plt.subplots()
        # Plot dart throws
        x = [i[1] for i in self.games_list[number]]
        y = [i[2] for i in self.games_list[number]]
        radii = [i[3] for i in self.games_list[number]]
        
        ax.scatter(x, y)
        x_length = len(x)
        for i in range(x_length):
            ax.annotate(str(i), (x[i], y[i]))
        
        # Draw circles
        circle1 = plt.Circle((0, 0), self.radius, fill=None)
        ax.add_artist(circle1)
        for i, r in enumerate(radii):
            new_circle = plt.Circle((0, 0), r, fill=None)
            ax.add_artist(new_circle)
            ax.annotate(str(i), (0, r))
            
        plt.title("Plotting game {} of {}".format(number, self.num_games))
        plt.xlim(-self.radius, self.radius)
        plt.ylim(-self.radius, self.radius)
        ax.set_aspect('equal')
        ax.set_facecolor('whitesmoke')
        plt.show()
        
        if save:
            plt.savefig("single_game_{}.png".format(number), dpi=150)
        return fig


if __name__ == "__main__":
    # Try argparse here
    game_parser = argparse.ArgumentParser(description='Options for playing dart game.')
    game_parser.add_argument('ngames', nargs='?', default=1000, type=int,
                             help='Specify number of games for simulation')
    game_parser.add_argument('coord', nargs='?', default=True, type=bool,
                             help='Randomly generate Cartesian or polar coordinates')
    args = game_parser.parse_args()
    
    print("Playing dart games...")
    game = DartSimulation(args.ngames, 1, args.coord)
    game.simulate()
    print("Done!")
    game.hist_games()
