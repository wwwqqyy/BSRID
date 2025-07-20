import subprocess


def run_eval(env_name: str) -> None:
    """
    Run the 'eval.py' script with specified arguments.

    Args:
        env_name (str): Name of the environment (e.g., "MsPacman").
    """
    ale_env_name = f"ALE/{env_name}-v5"
    run_name = f"{env_name}-100k-seed1"
    config_path = "config/BSRID.yaml"

    command = [
        "python",
        "-u",
        "eval.py",
        "-env_name", ale_env_name,
        "-run_name", run_name,
        "-config_path", config_path
    ]

    subprocess.call(command)


if __name__ == '__main__':
    # Example usage:
    game_list = [
        'Alien', 'Amidar', 
        'Assault', 'Asterix', 'BankHeist',
        'BattleZone', 'Boxing', 'Breakout', 
        'ChopperCommand', 'CrazyClimber', 'DemonAttack',
        'Freeway', 'Frostbite', 
        'Gopher', 'Hero', 'Jamesbond',
        'Kangaroo', 'Krull', 'KungFuMaster', 'MsPacman', 'Pong',
        'PrivateEye', 'Qbert', 'RoadRunner', 'Seaquest', 'UpNDown'
    ]
    # game_list = ['MsPacman']
    for env_game in game_list:
        run_eval(env_game)
