import numpy as np
import numpy.testing as npt
import pytest

from incoherence.experiments import (
    get_j,
    run_fold_posterior_into_reward,
    run_fold_posterior_into_reward_orig_mdp,
    run_increase_temp,
    run_retrain,
)
from incoherence.mdp import create_random_mdp, make_uniform_policy, stochastify
from incoherence.metrics import (
    compute_causal_forward_kl_divergence,
    compute_causal_forward_kl_divergence_belousov,
)
from incoherence.policy import boltzmann_rational_policy, compute_Vs_Qs
from incoherence.scenarios import create_two_cards_game


@pytest.mark.slow
def test_lemma_tempexp_fold_regression():
    mdp = create_random_mdp(2, 4, True)
    k = 5

    temp_exp_results = run_increase_temp(k, mdp=mdp, schedule=lambda n: [2**i for i in range(0, n)])
    fold_results = run_fold_posterior_into_reward(k, mdp=mdp)

    npt.assert_allclose(get_j(temp_exp_results), get_j(fold_results))


@pytest.mark.slow
def test_lemma_retrain_temp_regression():
    mdp = create_random_mdp(2, 5, True)
    k = 6

    retrain_results = run_retrain(k, mdp=mdp)[:-1]
    temp_results = run_increase_temp(k, mdp=mdp, schedule=lambda n: [i for i in range(1, n)])

    npt.assert_allclose(get_j(retrain_results), get_j(temp_results))


@pytest.mark.slow
def test_lemma_retrain_fold_det_regression():
    mdp = create_random_mdp(2, 4, True)
    k = 17

    retrain_results = run_retrain(k, mdp=mdp)
    retrain_results = [retrain_results[2 ** (i - 1)] for i in range(1, 5)]

    fold_results = run_fold_posterior_into_reward(k, mdp=mdp)
    fold_results = [fold_results[i] for i in range(1, 5)]

    npt.assert_allclose(get_j(retrain_results), get_j(fold_results))


@pytest.mark.slow
def test_lemma_retrain_fold_stoch_regression():
    mdp = create_random_mdp(2, 5, False)
    k = 17

    retrain_results = run_retrain(k, mdp=mdp)
    retrain_results = [retrain_results[2 ** (i - 1)] for i in range(1, 6)]

    fold_results = run_fold_posterior_into_reward(k, mdp=mdp)
    fold_results = [fold_results[i] for i in range(1, 6)]

    assert not np.allclose(get_j(retrain_results), get_j(fold_results))


@pytest.mark.slow
def test_lemma_retrain_fold_orig_regression():
    mdp = create_random_mdp(2, 4, False)
    k = 10

    retrain_results = run_retrain(k, mdp=mdp)
    fold_results = run_fold_posterior_into_reward_orig_mdp(k, mdp=mdp)

    npt.assert_allclose(get_j(retrain_results), get_j(fold_results))


@pytest.mark.slow
def test_strong_return_regression():
    mdp = create_random_mdp(2, 5, True)
    k = 6

    retrain_results = run_retrain(k, mdp=mdp)[:-1]
    js = get_j(retrain_results)

    assert all(x <= y for x, y in zip(js, js[1:])), "js is not monotonically nondecreasing"


@pytest.mark.slow
def test_alternative_kl_characterisation_regression():
    mdp = create_two_cards_game()
    mdp = stochastify(mdp)
    policy = make_uniform_policy(mdp)

    for _ in range(5):
        _, Q = compute_Vs_Qs(mdp, policy)
        br_policy = boltzmann_rational_policy(Q, temperature=1)
        npt.assert_almost_equal(
            compute_causal_forward_kl_divergence(mdp, policy, br_policy),
            compute_causal_forward_kl_divergence_belousov(mdp, policy, br_policy),
        )
        policy = br_policy
