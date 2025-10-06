from .distributions import P, bernoulli
from .mdp import MDP


def create_two_cards_game():
    # Define states, actions, transitions, and rewards for the two-cards game
    states = ["∅", "♦", "♥", "♦♦", "♦♥", "♥♦", "♥♥"]
    actions = {
        "∅": ["♦", "♥"],
        "♦": ["♦", "♥"],
        "♥": ["♦", "♥"],
        "♦♦": [" "],  # ' ' action to indicate the end of the game
        "♦♥": [" "],  # ' ' action to indicate the end of the game
        "♥♦": [" "],  # ' ' action to indicate the end of the game
        "♥♥": [" "],  # ' ' action to indicate the end of the game
        "end": [],
    }
    transitions = {
        "∅": {"♦": {"♦": 1.}, "♥": {"♥": 1.}},
        "♦": {"♦": {"♦♦": 1.}, "♥": {"♦♥": 1.}},
        "♥": {"♦": {"♥♦": 1.}, "♥": {"♥♥": 1.}},
        "♦♦": {" ": {"end": 1.}},  # ' ' action to indicate the end of the game
        "♦♥": {" ": {"end": 1.}},  # ' ' action to indicate the end of the game
        "♥♦": {" ": {"end": 1.}},  # ' ' action to indicate the end of the game
        "♥♥": {" ": {"end": 1.}},  # ' ' action to indicate the end of the game
    }
    transitions = {
        state: {action: P(distribution) for action, distribution in actions.items()}
        for state, actions in transitions.items()
    }
    rewards = {
        "∅": 1.,
        "♦": 1.,
        "♥": 1.,
        "♦♦": 1.,
        "♦♥": 0.001,
        "♥♦": 0.75,
        "♥♥": 0.33,
        "end": 1.,
    }

    state_time = {"∅": 0, "♦": 1, "♥": 1, "♦♦": 2, "♦♥": 2, "♥♦": 2, "♥♥": 2, "end": 3}

    rewards = {
        state: {action: bernoulli(prob) for action in actions[state]}
        for state, prob in rewards.items()
    }

    # Create the MDP object
    two_cards_game_mdp = MDP(
        states,
        actions,
        transitions,
        rewards,
        state_time=state_time,
        time_horizon=3,
        initial_state="∅",
    )
    return two_cards_game_mdp


def create_two_cards_game_stochastic_1():
    two_cards_game_mdp = create_two_cards_game()
    two_cards_game_mdp.transitions["♥"]["♥"] = P({"♥♥": 0.4, "♥♦": 0.6})  # Key difference
    return two_cards_game_mdp


def create_two_cards_game_stochastic_2():
    two_cards_game_mdp = create_two_cards_game()
    two_cards_game_mdp.transitions["♥"]["♥"] = P({"♥♥": 0.4, "♥♦": 0.6})  # Key difference
    two_cards_game_mdp.transitions["♦"]["♥"] = P({"♦♥": 0.3, "♦♦": 0.7})  # Key difference
    return two_cards_game_mdp


def create_two_cards_game_stochastic_3():
    two_cards_game_mdp = create_two_cards_game()
    two_cards_game_mdp.transitions["∅"]["♥"] = P({"♥": 0.8, "♦": 0.2})  # Key difference
    two_cards_game_mdp.transitions["♥"]["♥"] = P({"♥♥": 0.4, "♥♦": 0.6})  # Key difference
    two_cards_game_mdp.transitions["♦"]["♥"] = P({"♦♥": 0.3, "♦♦": 0.7})  # Key difference
    return two_cards_game_mdp


def create_two_cards_game_stochastic_4():
    two_cards_game_mdp = create_two_cards_game()
    two_cards_game_mdp.transitions["∅"]["♦"] = P({"♥": 0.1, "♦": 0.9})  # Key difference
    two_cards_game_mdp.transitions["∅"]["♥"] = P({"♥": 0.8, "♦": 0.2})  # Key difference
    two_cards_game_mdp.transitions["♥"]["♥"] = P({"♥♥": 0.4, "♥♦": 0.6})  # Key difference
    two_cards_game_mdp.transitions["♦"]["♥"] = P({"♦♥": 0.3, "♦♦": 0.7})  # Key difference
    return two_cards_game_mdp


def create_counterexample():
    # Define states, actions, transitions, and rewards for the two-cards game
    states = ["s0", "s1", "s2"]
    actions = {
        "s0": ["1", "2"],
        "s1": [" "],
        "s2": [" "],
        "end": [],
    }
    transitions = {
        "s0": {"1": {"s1": 3/4, "s2": 1/4}, "2": {"s1": 1/2, "s2": 1/2}},
        "s1": {" ": {"end": 1.}},
        "s2": {" ": {"end": 1.}}
    }
    transitions = {
        state: {action: P(distribution) for action, distribution in actions.items()}
        for state, actions in transitions.items()
    }
    rewards = {
        "s0": 1.,
        "s1": 1/3,
        "s2": 2/3,
        "end": 1.,
    }

    state_time = {"s0": 0, "s1": 1, "s2": 1, "end": 2}

    rewards = {
        state: {action: bernoulli(prob) for action in actions[state]}
        for state, prob in rewards.items()
    }

    # Create the MDP object
    mdp = MDP(
        states,
        actions,
        transitions,
        rewards,
        state_time=state_time,
        time_horizon=2,
        initial_state="s0",
    )
    return mdp


__all__ = [
    "create_counterexample",
    "create_two_cards_game",
    "create_two_cards_game_stochastic_1",
    "create_two_cards_game_stochastic_2",
    "create_two_cards_game_stochastic_3",
    "create_two_cards_game_stochastic_4",
]
