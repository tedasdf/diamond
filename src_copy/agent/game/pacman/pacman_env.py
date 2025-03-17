import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt

# Create the environment with render_mode set to "rgb_array"
env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')

# Reset the environment and get the initial observation
obs, info = env.reset()

# Print the shape and dtype of the observation
print(f"Observation Shape: {obs.shape}")
print(f"Observation Data Type: {obs.dtype}")

# Display the image
plt.imshow(obs)
plt.axis('off')  # Hide axes
plt.show()


# # Run the environment
# for _ in range(1000):
#     # Render the environment to show the game window
#     env.render()

#     # Default action: No-op (action 0)
#     action = 0

#     # Check for keyboard input to control the game
#     if keyboard.is_pressed('left'):
#         action = key_action_mapping['left']
#     elif keyboard.is_pressed('right'):
#         action = key_action_mapping['right']
#     elif keyboard.is_pressed('space'):
#         action = key_action_mapping['space']
#     elif keyboard.is_pressed('down'):
#         action = key_action_mapping['down']
    

#     # Step through the environment with the chosen action
#     obs, reward, terminated, truncated, info = env.step(action)



#     # Reset the environment if the game ends
#     if terminated or truncated:
#         obs, info = env.reset()

# # Close the environment after the loop
# env.close()
