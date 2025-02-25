from Bots.bot_ai import BotAI
from Bots.first_ai import FirstAI
from Bots.wingmandmd import WingmanDMD
from Bots.rl_agent import RLAgent

ai_bots = {
    "FirstAI": FirstAI,
    "WingmanDMD": WingmanDMD,
    "RLAgent": RLAgent,
}


def ai_factory(bot_selection="RLAgent") -> BotAI:
    return ai_bots[bot_selection]()
