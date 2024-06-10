import random
# from logger import logger


def monte_carlo_eval(prompt):
  
        # Simulating different types of responses
        response_types = ['highly relevant', 'somewhat relevant', 'irrelevant']
        scores = {'highly relevant': 3, 'somewhat relevant': 2, 'irrelevant': 1}

        # Perform multiple random trials
        trials = 100
        total_score = 0
        for _ in range(trials):
            response = random.choice(response_types)
            total_score += scores[response]

        # Average score represents the evaluation
        return total_score / trials
    
   


def elo_eval(prompt, base_rating=1500):
   

        # Simulate the outcome of the prompt against standard criteria
        # Here, we randomly decide if the prompt 'wins', 'loses', or 'draws'
        outcomes = ['win', 'loss', 'draw']
        outcome = random.choice(outcomes)

        # Elo rating formula parameters
        K = 30  # Maximum change in rating
        R_base = 10 ** (base_rating / 400)
        R_opponent = 10 ** (1600 / 400)  # Assuming a fixed opponent rating
        expected_score = R_base / (R_base + R_opponent)

        # Calculate the new rating based on the outcome
        actual_score = {'win': 1, 'loss': 0, 'draw': 0.5}[outcome]
        new_rating = base_rating + K * (actual_score - expected_score)
    
        return new_rating
    
 


def elo_ratings_func(prompts, elo_ratings, K=30, opponent_rating=1600):
    """
    Update Elo ratings for a list of prompts based on simulated outcomes.

    Parameters:
    prompts (list): List of prompts to be evaluated.
    elo_ratings (dict): Current Elo ratings for each prompt.
    K (int): Maximum change in rating.
    opponent_rating (int): Fixed rating of the opponent for simulation.

    Returns:
    dict: Updated Elo ratings.
    """

    for prompt in prompts:
        # Simulate an outcome against the standard criteria or another prompt
        outcome = random.choice(['win', 'loss', 'draw'])

        # Calculate the new rating based on the outcome
        actual_score = {'win': 1, 'loss': 0, 'draw': 0.5}[outcome]
        R_base = 10 ** (elo_ratings[prompt] / 400)
        R_opponent = 10 ** (opponent_rating / 400)
        expected_score = R_base / (R_base + R_opponent)
        elo_ratings[prompt] += K * (actual_score - expected_score)

    return elo_ratings


def evaluate_prompt(main_prompt, test_cases):
    evaluations = {}

    # Evaluate the main prompt using Monte Carlo and Elo methods
    evaluations['main_prompt'] = {
        'Monte Carlo Evaluation': monte_carlo_eval(main_prompt),
        'Elo Rating Evaluation': elo_eval(main_prompt)
    }

    # Evaluate each test case
    for idx, test_case in enumerate(test_cases):
        evaluations[f'test_case_{idx+1}'] = {
            'Monte Carlo Evaluation': monte_carlo_eval(test_case),
            'Elo Rating Evaluation': elo_eval(test_case)
        }

    return evaluations