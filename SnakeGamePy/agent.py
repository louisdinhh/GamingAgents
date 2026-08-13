import torch
import random
import numpy as np
from collections import deque

from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# Exploration settings
EPSILON_START = 80
EPSILON_MIN = 5

# Prevent the snake from wandering forever
MAX_STEPS_WITHOUT_FOOD = 100


class Agent:

    def __init__(self):
        self.n_games = 0

        # Exploration
        self.epsilon = EPSILON_START

        # Discount factor
        self.gamma = 0.9

        self.memory = deque(maxlen=MAX_MEMORY)

        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(
            self.model,
            lr=LR,
            gamma=self.gamma
        )

        # Used to detect loops
        self.state_history = deque(maxlen=30)

    def get_state(self, game):

        head = game.snake[0]

        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [

            # Danger straight
            (
                (dir_r and game.is_collision(point_r)) or
                (dir_l and game.is_collision(point_l)) or
                (dir_u and game.is_collision(point_u)) or
                (dir_d and game.is_collision(point_d))
            ),

            # Danger right
            (
                (dir_u and game.is_collision(point_r)) or
                (dir_d and game.is_collision(point_l)) or
                (dir_l and game.is_collision(point_u)) or
                (dir_r and game.is_collision(point_d))
            ),

            # Danger left
            (
                (dir_d and game.is_collision(point_r)) or
                (dir_u and game.is_collision(point_l)) or
                (dir_r and game.is_collision(point_u)) or
                (dir_l and game.is_collision(point_d))
            ),

            # Current direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < head.x,   # food left
            game.food.x > head.x,   # food right
            game.food.y < head.y,   # food up
            game.food.y > head.y,   # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )

    def train_long_memory(self):

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory,
                BATCH_SIZE
            )
        else:
            mini_sample = self.memory

        if len(mini_sample) == 0:
            return

        states, actions, rewards, next_states, dones = zip(
            *mini_sample
        )

        self.trainer.train_step(
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def train_short_memory(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        self.trainer.train_step(
            state,
            action,
            reward,
            next_state,
            done
        )

    def get_action(self, state):

        # Slowly reduce exploration, but NEVER below EPSILON_MIN
        self.epsilon = max(
            EPSILON_MIN,
            EPSILON_START - self.n_games
        )

        final_move = [0, 0, 0]

        # Exploration
        if random.randint(0, 200) < self.epsilon:

            move = random.randint(0, 2)
            final_move[move] = 1

        # Exploitation
        else:

            state0 = torch.tensor(
                state,
                dtype=torch.float
            )

            with torch.no_grad():
                prediction = self.model(state0)

            move = torch.argmax(prediction).item()

            final_move[move] = 1

        return final_move

    def reset_loop_detection(self):
        self.state_history.clear()

    def is_looping(self, state):

        # Convert numpy array to tuple so it can be stored
        state_tuple = tuple(state.tolist())

        if state_tuple in self.state_history:
            return True

        self.state_history.append(state_tuple)

        return False


def train():

    plot_scores = []
    plot_mean_scores = []

    total_score = 0
    record = 0

    agent = Agent()
    game = SnakeGameAI()

    steps_without_food = 0

    while True:

        # -----------------------------
        # OLD STATE
        # -----------------------------
        state_old = agent.get_state(game)

        # -----------------------------
        # GET ACTION
        # -----------------------------
        final_move = agent.get_action(state_old)

        # Save old distance to food
        head_old = game.snake[0]

        old_distance = abs(
            head_old.x - game.food.x
        ) + abs(
            head_old.y - game.food.y
        )

        # -----------------------------
        # PERFORM MOVE
        # -----------------------------
        reward, done, score = game.play_step(final_move)

        # -----------------------------
        # NEW STATE
        # -----------------------------
        state_new = agent.get_state(game)

        # -----------------------------
        # REWARD SHAPING
        # -----------------------------

        if not done:

            head_new = game.snake[0]

            new_distance = abs(
                head_new.x - game.food.x
            ) + abs(
                head_new.y - game.food.y
            )

            # Reward moving toward food
            if new_distance < old_distance:
                reward += 0.1

            # Small penalty for moving away
            elif new_distance > old_distance:
                reward -= 0.1

        # -----------------------------
        # LOOP DETECTION
        # -----------------------------

        if agent.is_looping(state_new):

            # Penalize repeated states
            reward -= 0.2

        # -----------------------------
        # COUNT STEPS
        # -----------------------------

        steps_without_food += 1

        # If food was eaten, reset the counter
        if score > getattr(agent, "last_score", 0):
            steps_without_food = 0

        agent.last_score = score

        # Kill the game if snake wanders too long
        if steps_without_food >= MAX_STEPS_WITHOUT_FOOD:

            reward = -10
            done = True

        # -----------------------------
        # SHORT-TERM TRAINING
        # -----------------------------

        agent.train_short_memory(
            state_old,
            final_move,
            reward,
            state_new,
            done
        )

        # -----------------------------
        # REMEMBER
        # -----------------------------

        agent.remember(
            state_old,
            final_move,
            reward,
            state_new,
            done
        )

        # -----------------------------
        # GAME OVER
        # -----------------------------

        if done:

            game.reset()

            agent.n_games += 1

            agent.reset_loop_detection()

            steps_without_food = 0
            agent.last_score = 0

            # Long-term training
            agent.train_long_memory()

            # Save best model
            if score > record:
                record = score

                # Uncomment if your model has save()
                # agent.model.save()

            print(
                f"Game: {agent.n_games} "
                f"Score: {score} "
                f"Record: {record} "
                f"Epsilon: {agent.epsilon:.1f}"
            )

            # Plot
            plot_scores.append(score)

            total_score += score

            mean_score = total_score / agent.n_games

            plot_mean_scores.append(mean_score)

            plot(
                plot_scores,
                plot_mean_scores
            )


if __name__ == "__main__":
    train()
