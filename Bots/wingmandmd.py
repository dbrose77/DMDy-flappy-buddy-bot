from Bots.bot_ai import BotAI
from Bots.data import PlayState


class WingmanDMD(BotAI):
    SCREEN_HEIGHT = 512
    fly = True

    def __init__(self):
        self.name = "WingmanDMD"
        self.token_id = None

    def find_closest_coin(self, current_game_state: PlayState):
        coins = [obs for obs in current_game_state.obstacles if obs.type == 'Coin']
        if not coins:
            return None
        
        # Find the leftmost (closest) coin that's ahead of the player
        closest_coin = None
        min_distance = float('inf')
        player_x = current_game_state.player.pos_x
        
        for coin in coins:
            if coin.origin_x > player_x:  # Only consider coins ahead of the player
                distance = coin.origin_x - player_x
                if distance < min_distance:
                    min_distance = distance
                    closest_coin = coin
        
        return closest_coin

    def is_path_dangerous(self, current_game_state: PlayState, target_y: float) -> bool:
        player = current_game_state.player
        player_x = player.pos_x
        player_y = player.pos_y
        
        # Define the path area we're checking
        path_start_y = min(player_y, target_y)
        path_end_y = max(player_y, target_y)
        
        # Look for dangerous obstacles ahead
        for obstacle in current_game_state.obstacles:
            if obstacle.type == 'Coin':  # Coins are not dangerous
                continue
                
            # Only check obstacles ahead of us
            if obstacle.origin_x <= player_x:
                continue
                
            # Check if obstacle is too close (within next 200 pixels)
            if obstacle.origin_x - player_x > 200:
                continue
                
            # Check if obstacle intersects with our intended path
            obstacle_top = obstacle.origin_y
            obstacle_bottom = obstacle.origin_y + obstacle.height
            
            # If the obstacle's vertical range overlaps with our path
            if (obstacle_top <= path_end_y and obstacle_bottom >= path_start_y):
                return True
                
        return False

    def play(self, current_game_state: PlayState):
        closest_coin = self.find_closest_coin(current_game_state)
        
        if closest_coin is None:
            return self.fly

        player_center_y = current_game_state.player.pos_y + (current_game_state.player.height / 2)
        coin_center_y = closest_coin.origin_y + (closest_coin.height / 2)
        
        if self.is_path_dangerous(current_game_state, coin_center_y):
            screen_height = self.SCREEN_HEIGHT
            if player_center_y > screen_height / 2:
                self.fly = False
            else:
                self.fly = True
            return self.fly
        
        if player_center_y > coin_center_y:
            self.fly = True
        elif player_center_y < coin_center_y:
            self.fly = False
        
        return self.fly

    def get_name(self):
        return self.name
