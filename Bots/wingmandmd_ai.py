from Bots.bot_ai import BotAI
from Bots.data import PlayState


class WingmanDMDAI(BotAI):
    fly = True

    def play(self, current_game_state: PlayState):
        print(f"Current game state: {current_game_state}")
        if current_game_state.player.pos_y > 450:
            self.fly = True

        if current_game_state.player.pos_y < 50:
            self.fly = False

        return self.fly

    def get_name(self):
        return self.name

    def __init__(self):
        self.name = "WingmanDMD"
        self.token_id = None
