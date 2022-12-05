from game import Game
from config import MuZeroConfig, MinMaxStats, TestResults, TrainResults
from replay import ReplayBuffer
from networks import CartPoleNetwork, train_network
from mcts import Node
from mcts import expand_root, add_exploration_noise, run_mcts
from mcts import select_action, backpropagate
from tensorflow.keras.optimizers import Adam

CARTPOLE_STOP_REWARD = 195
LAST_N = 40


def self_play(env, config: MuZeroConfig, replay_buffer: ReplayBuffer, network: CartPoleNetwork):
    # Create optimizer for training
    optimizer = Adam(learning_rate=config.lr_init)
    games_played = 0
    test_rewards = TestResults()
    train_results = TrainResults()
    for i in range(config.num_epochs):  # Number of Steps of train/play alternations
        print(f"Epoch Number {i}")
        games_played, score = play_games(
            config, replay_buffer, network, env, games_played)
        print("Train score:", score)
        train_network(config, network, replay_buffer, optimizer, train_results)
        print("Eval score:", test(config, network, env, test_rewards))
        reward_avg, n = test_rewards.last_n_average(LAST_N)
        print(f"Last {n} rewards average: {reward_avg}")
        if reward_avg >= CARTPOLE_STOP_REWARD:
            print("Agent Successfully learned cartpole")
            # Plotting code here
            train_results.plot_individual_losses()
            train_results.plot_total_loss()
            test_rewards.plot_rewards()
            return


def play_games(config: MuZeroConfig, replay_buffer: ReplayBuffer, network: CartPoleNetwork, env, games_played):
    returns = 0

    for _ in range(config.games_per_epoch):
        game = play_game(config, network, env, games_played)
        replay_buffer.save_game(game)
        returns += sum(game.reward_history)
        games_played += 1

    return games_played, returns / config.games_per_epoch


def play_game(config: MuZeroConfig, network: CartPoleNetwork, env, games_played):
    """
    Plays one game of environment
    config: configurations for muzero
    network: network model used in MCTS unrolling
    env: the Gym environment
    games_played: how many games played, used in visit_softmax_temperature_fn
    """
    # env.seed(1) Use for reproducibility of trajectories
    start_state = env.reset()
    # Create Game Objects
    game = Game(config.action_space_size, config.discount, start_state)
    while not game.done and len(game.action_history) < config.max_moves:
        # Min Max Stats for child selection in tree (normalized Q-values)
        min_max_stats = MinMaxStats(config.known_bounds)
        curr_state = game.curr_state
        root = Node(0)
        # Expand root and backpropagate once
        value = expand_root(root, list(range(config.action_space_size)),
                            network, curr_state)
        backpropagate([root], value, config.discount, min_max_stats)
        add_exploration_noise(config, root)
        # Run MCTS
        run_mcts(config, root, network, min_max_stats)
        # Select action from root
        action = select_action(config, games_played, root, network)
        game.action(action, env)
        # Take action and store tree statistics
        game.store_search_statistics(root)
    print(f'Total reward for game: {sum(game.reward_history)}')
    return game


def test(config: MuZeroConfig, network: CartPoleNetwork, env, test_rewards: TestResults):
    """
    Using a trained network_model, test games
    config: configurations for muzero
    network: The network model will be used for inference to conduct MCTS
    game_list (list[Game]): The list of games that were played by the network_model
    """

    print('\n=========== TESTING ===========')
    returns = 0
    for _ in range(config.episodes_per_test):
        # env.seed(1) Use for reproducibility of trajectories
        start_state = env.reset()
        game = Game(config.action_space_size, config.discount, start_state)
        while not game.done and len(game.action_history) < config.max_moves:
            min_max_stats = MinMaxStats(config.known_bounds)
            curr_state = game.curr_state
            root = Node(0)
            value = expand_root(root, list(range(config.action_space_size)),
                                network, curr_state)
            backpropagate([root], value, config.discount, min_max_stats)
            # Run MCTS
            run_mcts(config, root, network, min_max_stats)
            action = select_action(config, len(
                game.action_history), root, network, test=True)
            game.action(action, env)
        total_reward = sum(game.reward_history)
        print(f'Total reward for game: {total_reward}')
        test_rewards.add_reward(total_reward)
        returns += total_reward
    return returns / config.episodes_per_test
