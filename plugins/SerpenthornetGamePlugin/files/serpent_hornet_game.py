from serpent.game import Game

from .api.api import hornetAPI

from serpent.utilities import Singleton




class SerpenthornetGame(Game, metaclass=Singleton):

    def __init__(self, **kwargs):
        kwargs["platform"] = "steam"

        kwargs["window_name"] = "Golden Hornet"

        kwargs["app_id"] = "739260"
        kwargs["app_args"] = None

        super().__init__(**kwargs)

        self.api_class = hornetAPI
        self.api_instance = None

        # self.frame_transformation_pipeline_string = "RESIZE:100x100|GRAYSCALE|FLOAT"
        self.frame_transformation_pipeline_string = "RESIZE:100x100|JPG|FLOAT"
        self.frame_width = 100
        self.frame_height = 100
        # self.frame_channels = 3

    @property
    def screen_regions(self):
        regions = {
            "SCORE": (35, 7, 65, 87),
            "GAME_STATE": (87, 343, 330, 650),
            "HUD_HEART_1": (519, 22, 525, 30),
            "HUD_HEART_2": (519, 30, 525, 38),
            "HUD_HEART_3": (519, 38, 525, 46),
            "HUD_HEART_4": (519, 46, 525, 54),
            "HUD_HEART_5": (519, 54, 525, 62),
            "HUD_HEART_6": (519, 62, 525, 70),
            "HUD_HEART_7": (519, 70, 525, 78),
            "HUD_HEART_8": (519, 78, 525, 86),
            "HUD_HEART_9": (519, 86, 525, 94),
            "HUD_HEART_10": (519, 94, 525, 102),
            "HUD_HEART_11": (519, 102, 525, 110),
            "HUD_HEART_12": (519, 110, 525, 118),
            "HUD_HEART_13": (519, 118, 525, 126),
            "HUD_HEART_14": (519, 126, 525, 134),
            "HUD_HEART_15": (519, 134, 525, 142)
        }

        return regions

    @property
    def ocr_presets(self):
        presets = {
            "SAMPLE_PRESET": {
                "extract": {
                    "gradient_size": 1,
                    "closing_size": 1
                },
                "perform": {
                    "scale": 10,
                    "order": 1,
                    "horizontal_closing": 1,
                    "vertical_closing": 1
                }
            }
        }

        return presets
