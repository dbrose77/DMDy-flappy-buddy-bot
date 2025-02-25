import numpy as np
import tensorflow as tf
from collections import deque
import random
from Bots.bot_ai import BotAI
from Bots.data import PlayState
import time
import os
import datetime
import threading
from queue import Queue
import logging
import pickle

# tensorboard --logdir=Bots/saved_model/logs

class RLAgent(BotAI):
    def __init__(self):
        self.name = "RLAgent"
        
        # Create save directory if it doesn't exist
        self.save_dir = os.path.join(os.path.dirname(__file__), 'saved_model')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # RL parameters
        self.state_size = 7
        self.action_size = 2
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.997
        self.learning_rate = 0.001
        
        # Game state tracking
        self.previous_state = None
        self.previous_action = None
        self.previous_score = 0
        
        # Training metrics
        self.episode_count = 0
        self.current_episode_reward = 0
        
        # Add prediction caching
        self.state_cache = {}
        self.cache_size = 1000
        
        # Add training control
        self.training_interval = 0.5
        self.last_training_time = time.time()
        self.batch_size = 32
        self.last_save_time = time.time()
        self.save_interval = 300  # Save every 5 minutes
        
        # Add prediction queue
        self.prediction_queue = Queue(maxsize=1)
        self.prediction_result = None
        
        # Optimize model for inference
        self.model = self._build_model()
        
        # Load saved data if it exists
        self._load_training_data()
        
        # Start threads
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.prediction_thread = threading.Thread(target=self._prediction_loop, daemon=True)
        self.training_thread.start()
        self.prediction_thread.start()
        
        self._is_shutting_down = False
        
        # Add TensorBoard logging
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(self.save_dir, 'logs', current_time)
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)
        print(f"TensorBoard log directory: {self.log_dir}")

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', 
                     optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                     run_eagerly=False)  # Disable eager execution for better performance
        return model

    def _get_state(self, game_state: PlayState):
        # Cache frequently accessed values
        player_x = game_state.player.pos_x
        player_y = game_state.player.pos_y
        
        state = np.zeros(self.state_size, dtype=np.float32)
        # Normalize y position with more granularity
        state[0] = (player_y - 250) / 250.0  # Center at 0, range from -1 to 1
        state[1] = game_state.player.rotation / 360.0

        closest_coin_dist = float('inf')
        closest_enemy_dist = float('inf')
        closest_coin_dx = 0
        closest_coin_dy = 0
        closest_enemy_dx = 0
        closest_enemy_dy = 0
        enemy_close_area = 0  # Initialize this variable
        
        # Process coins and enemies separately for better efficiency
        coins = [obs for obs in game_state.obstacles if obs.type == 'Coin']
        enemies = [obs for obs in game_state.obstacles if obs.type != 'Coin']
        
        if coins:
            # Vectorized distance calculation for coins
            coin_dxs = np.array([coin.origin_x - player_x for coin in coins])
            coin_dys = np.array([coin.origin_y - player_y for coin in coins])
            coin_dists = np.sqrt(coin_dxs**2 + coin_dys**2)
            min_coin_idx = np.argmin(coin_dists)
            closest_coin_dx = coin_dxs[min_coin_idx]
            closest_coin_dy = coin_dys[min_coin_idx]
            
        if enemies:
            # Vectorized distance calculation for enemies
            enemy_dxs = np.array([enemy.origin_x - player_x for enemy in enemies])
            enemy_dys = np.array([enemy.origin_y - player_y for enemy in enemies])
            enemy_dists = np.sqrt(enemy_dxs**2 + enemy_dys**2)
            min_enemy_idx = np.argmin(enemy_dists)
            closest_enemy_dx = enemy_dxs[min_enemy_idx]
            closest_enemy_dy = enemy_dys[min_enemy_idx]
            enemy_close_area = 1.0 if enemy_dists[min_enemy_idx] < enemies[min_enemy_idx].close_area_width else 0.0

        state[2] = closest_coin_dx / 1000.0
        state[3] = closest_coin_dy / 500.0
        state[4] = closest_enemy_dx / 1000.0
        state[5] = closest_enemy_dy / 500.0
        state[6] = enemy_close_area

        return state

    def _prediction_loop(self):
        while True:
            try:
                if not self.prediction_queue.empty():
                    state = self.prediction_queue.get()
                    state_tuple = tuple(state)
                    
                    if state_tuple in self.state_cache:
                        self.prediction_result = self.state_cache[state_tuple]
                    else:
                        if random.random() <= self.epsilon:
                            action = random.randint(0, 1)
                        else:
                            act_values = self.model.predict(state.reshape(1, -1), verbose=0)
                            action = int(np.argmax(act_values[0]))
                        
                        self.state_cache[state_tuple] = action
                        if len(self.state_cache) > self.cache_size:
                            self.state_cache.pop(next(iter(self.state_cache)))
                        
                        self.prediction_result = action
                time.sleep(0.001)  # Small sleep to prevent CPU overload
            except Exception as e:
                logging.error(f"Error in prediction loop: {e}")

    def _act(self, state):
        # Put state in prediction queue
        if self.prediction_queue.empty():
            self.prediction_queue.put(state)
        
        # Return last prediction result or random action if no prediction available
        if self.prediction_result is not None:
            result = self.prediction_result
            self.prediction_result = None
            return result
        return random.randint(0, 1)

    def _remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def _save_training_data(self):
        try:
            # Save model weights with correct extension
            weights_path = os.path.join(self.save_dir, 'model.weights.h5')
            self.model.save_weights(weights_path)
            
            # Save other training data
            training_data = {
                'memory': list(self.memory),
                'epsilon': self.epsilon,
                'episode_count': self.episode_count
            }
            
            with open(os.path.join(self.save_dir, 'training_data.pkl'), 'wb') as f:
                pickle.dump(training_data, f)
                
            print(f"\nSaved training data. Episodes: {self.episode_count}, Epsilon: {self.epsilon:.3f}")
            
        except Exception as e:
            logging.error(f"Error saving training data: {e}")

    def _load_training_data(self):
        try:
            # Load model weights if they exist
            weights_path = os.path.join(self.save_dir, 'model.weights.h5')
            if os.path.exists(weights_path):
                self.model.load_weights(weights_path)
                
            # Load other training data
            data_path = os.path.join(self.save_dir, 'training_data.pkl')
            if os.path.exists(data_path):
                with open(data_path, 'rb') as f:
                    training_data = pickle.load(f)
                    
                self.memory = deque(training_data['memory'], maxlen=10000)
                self.epsilon = training_data['epsilon']
                self.episode_count = training_data['episode_count']
                
                print(f"\nLoaded training data. Episodes: {self.episode_count}, Epsilon: {self.epsilon:.3f}")
                
        except Exception as e:
            logging.error(f"Error loading training data: {e}")

    def _log_metrics(self, episode_reward, score, epsilon, loss=None):
        """Log metrics to TensorBoard"""
        with self.summary_writer.as_default():
            tf.summary.scalar('Episode Reward', episode_reward, step=self.episode_count)
            tf.summary.scalar('Score', score, step=self.episode_count)
            tf.summary.scalar('Epsilon', epsilon, step=self.episode_count)
            if loss is not None:
                tf.summary.scalar('Loss', loss, step=self.episode_count)

    def _training_loop(self):
        while True:
            current_time = time.time()
            
            # Check if it's time to save
            if current_time - self.last_save_time >= self.save_interval:
                self._save_training_data()
                self.last_save_time = current_time
            
            if current_time - self.last_training_time >= self.training_interval and len(self.memory) >= self.batch_size:
                minibatch = random.sample(self.memory, self.batch_size)
                
                # Batch process all states at once
                states = np.vstack([x[0] for x in minibatch])
                next_states = np.vstack([x[3] for x in minibatch])
                
                current_qs = self.model.predict(states, verbose=0)
                next_qs = self.model.predict(next_states, verbose=0)
                
                # Vectorized Q-value updates
                for i, (_, action, reward, _) in enumerate(minibatch):
                    current_qs[i][action] = reward + self.gamma * np.max(next_qs[i])
                
                # Train and get loss
                history = self.model.fit(states, current_qs, epochs=1, verbose=0, batch_size=self.batch_size)
                loss = history.history['loss'][0]
                
                # Log training metrics
                with self.summary_writer.as_default():
                    tf.summary.scalar('Training Loss', loss, step=self.episode_count)
                    tf.summary.scalar('Memory Size', len(self.memory), step=self.episode_count)
                
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
                    
                self.last_training_time = current_time
                
            time.sleep(0.1)

    def _calculate_reward(self, current_game_state: PlayState, previous_score: int):
        """Calculate detailed reward based on game state"""
        reward = 0
        
        # Base reward from score difference (collecting coins)
        reward += (current_game_state.score - previous_score) * 2
        
        # Position-based rewards
        player_y = current_game_state.player.pos_y
        
        # Stronger penalties for extreme positions
        if player_y < 100:  # too low
            reward -= 200 * (1 - player_y/100)  # Progressive penalty that increases as y approaches 0
        elif player_y > 400:  # too high
            reward -= 200 * ((player_y-400)/100)  # Progressive penalty that increases as y approaches 500
        
        # Reward for maintaining "safe" altitude
        if 200 <= player_y <= 300:  # Optimal zone
            reward += 10
        elif 100 <= player_y <= 400:  # Acceptable zone
            reward += 5
        
        # Coin proximity reward with vertical position consideration
        closest_coin_dist = float('inf')
        for obstacle in current_game_state.obstacles:
            if obstacle.type == 'Coin':
                dx = obstacle.origin_x - current_game_state.player.pos_x
                dy = obstacle.origin_y - player_y
                dist = (dx * dx + dy * dy) ** 0.5
                closest_coin_dist = min(closest_coin_dist, dist)
        
        # Scale coin proximity reward based on vertical position
        if closest_coin_dist != float('inf'):
            coin_proximity_reward = 25 / (closest_coin_dist + 1)
            # Reduce reward if in bad vertical position
            if player_y < 100 or player_y > 400:
                coin_proximity_reward *= 0.5
            reward += coin_proximity_reward * 100
        
        # Enemy avoidance reward
        for obstacle in current_game_state.obstacles:
            if obstacle.type != 'Coin':
                dx = obstacle.origin_x - current_game_state.player.pos_x
                dy = obstacle.origin_y - player_y
                dist = (dx * dx + dy * dy) ** 0.5
                
                if dist < obstacle.close_area_width:
                    reward -= 75  # Increased penalty for being too close
                elif dist < obstacle.close_area_width * 1.5:
                    reward -= 150  # Smaller penalty for being somewhat close
                    
        # Terminal state rewards
        if current_game_state.player.state == "died":
            reward -= 500  # Increased death penalty
        elif current_game_state.player.state == "finished":
            reward += 1000  # Increased level completion reward
            
        print(f"Reward: {reward}")
        return reward

    def play(self, current_game_state: PlayState):
        try:
            current_state = self._get_state(current_game_state)
            
            # Process rewards and remember
            if self.previous_state is not None:
                reward = self._calculate_reward(current_game_state, self.previous_score)
                self.current_episode_reward += reward
                self._remember(self.previous_state, self.previous_action, reward, current_state)

                # Handle episode end immediately
                if current_game_state.player.state in ["died", "finished"]:
                    print(f"\nEpisode {self.episode_count + 1} ended: {current_game_state.player.state}")
                    print(f"Score: {current_game_state.score}")
                    print(f"Total Reward: {self.current_episode_reward}")
                    print(f"Epsilon: {self.epsilon:.3f}")
                    
                    # Log episode metrics
                    self._log_metrics(
                        episode_reward=self.current_episode_reward,
                        score=current_game_state.score,
                        epsilon=self.epsilon
                    )
                    
                    self.current_episode_reward = 0
                    self.episode_count += 1

            # Choose action
            action = self._act(current_state)
            
            # Save current state
            self.previous_state = current_state
            self.previous_action = action
            self.previous_score = current_game_state.score
            
            return bool(action)
            
        except Exception as e:
            logging.error(f"Error in play loop: {e}")
            return False

    def get_name(self):
        return self.name 

    def shutdown(self):
        """Clean up and save the model before shutting down"""
        if not self._is_shutting_down:
            self._is_shutting_down = True
            print("\nShutting down RLAgent - Saving training data...")
            self._save_training_data() 