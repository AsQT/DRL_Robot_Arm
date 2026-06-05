
def Get_Environment_ID(name: str, env_mode: int) -> str:
    """
    Description:
        Get a string of the gym environment ID for parameters defined by the function input.

    Args:
        (1) name [string]: Name of the robotic structure.
        (2) env_mode [int]: The name of the environment mode.
                                env_mode = 'Default' or 'Collision-Free'

    Returns:
        (1) parameter [string]: The string of the desired gym environment ID.
    """

    try:
        assert env_mode in ['Default', 'Collision-Free']

        return {'YASKAWA_GP7': lambda env_m: f'Yaskawagp7-{env_m}-Reach-v0'
        }[name](env_mode)
    
    except AssertionError as error:
        print(f'[ERROR] Information: {error}')
        print('[ERROR] Incorrect environment mode selected. The selected mode must be chosen from the two options (Default, Collision-Free).')