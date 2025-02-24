from Bots.bot_ai import BotAI
from Bots.first_ai import FirstAI
from Bots.wingmandmd_ai import WingmanDMDAI

ai_bots = {
    "FirstAI": FirstAI,
    "WingmanDMD": WingmanDMDAI,
}


def ai_factory(bot_selection="WingmanDMD") -> BotAI:
    return ai_bots[bot_selection]()
