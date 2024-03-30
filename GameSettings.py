import json
import os

class GameSettings:
    def __init__(self, settings_file="game_settings.json"):
        self.settings_file = settings_file
        # Define default settings as a class variable or inside __init__
        self.default_settings = {
            "music_enabled": True,
            "render2D": True,
            "game_scale": 2
        }
        self.settings = self.load_settings()

    def load_settings(self):
        # Load settings from a file, or return default settings if file doesn't exist
        if os.path.exists(self.settings_file):
            with open(self.settings_file, "r") as file:
                loaded_settings = json.load(file)
                # Merge loaded settings with default settings
                # Default settings will only be used if the corresponding key is missing in loaded_settings
                return {**self.default_settings, **loaded_settings}
        else:
            # Return default settings if the settings file does not exist
            return self.default_settings

    def save_settings(self):
        # Save the current settings to a file
        with open(self.settings_file, "w") as file:
            json.dump(self.settings, file, indent=4)

    def get_setting(self, setting, default=None):
        # Get a setting value or a default if it's not set
        return self.settings.get(setting, default)

    def set_setting(self, setting, value):
        # Update a setting and save the file
        self.settings[setting] = value
        self.save_settings()
