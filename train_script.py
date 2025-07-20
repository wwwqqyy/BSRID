import subprocess


def run_training(env_name: str):
    """
    Run the training process for the specified environment with the given configuration.

    Parameters:
    env_name (str): Name of the Atari game environment (e.g., "Pong").
    """

    # Construct the command based on the provided environment name and other parameters
    command = [
        "python3",
        "-u",
        "train.py",
        "-n",
        f"{env_name}-100k-seed1",
        "-seed",
        "1",
        "-config_path",
        "config/BSRID.yaml",
        "-env_name",
        f"ALE/{env_name}-v5",
    ]

    # Execute the command using subprocess.run
    subprocess.run(command, check=True)


if __name__ == '__main__':
    # Example usage
    # run_training("Pong")  # Train on Pong environment with default seed (1)
    # run_training("Breakout", seed=42)  # Train on Breakout environment with seed 42
    game_list = [
        # 'Alien', 'Amidar', 
        # 'Assault', 'Asterix', 'BankHeist',
        # 'BattleZone', 'Boxing', 'Breakout', 
        'ChopperCommand', 'CrazyClimber', 'DemonAttack',
        'Freeway', 'Frostbite', 
        # 'Gopher', 'Hero', 'Jamesbond',
        # 'Kangaroo', 'Krull', 'KungFuMaster', 'MsPacman', 'Pong',
        # 'PrivateEye', 'Qbert', 'RoadRunner', 'Seaquest', 'UpNDown'
    ]
    # game_list = ['MsPacman']
    for env_game in game_list:
        run_training(env_game)
