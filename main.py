import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import torch
from torch import nn
import torch.nn.functional as F
import pygame

from SolsticeGame import SolsticeGame

# Check if CUDA (GPU support) is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")
if (torch.cuda.is_available()):
    print(torch.cuda.is_available())  # Should return True if CUDA is properly set up
    print(torch.cuda.current_device())  # Shows the current CUDA device ID
    print(torch.cuda.device_count())  # Shows the number of available CUDA devices
    print(torch.cuda.get_device_name(0))  # Shows the name of the CUDA device, change 0 accordingly if multiple GPUs


class DQN(nn.Module):
    def __init__(self, in_channels, map_height, map_width, h1_nodes, out_actions):
        super().__init__()

        # Calculate the flattened input size
        self.flattened_size = in_channels * map_height * map_width

        # Define network layers
        self.fc1 = nn.Linear(self.flattened_size, h1_nodes)  # first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions)  # ouptut layer w

    def forward(self, x):
        # Ensure x is properly flattened
        # TODO: only if multi
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))  # Apply rectified linear unit (ReLU) activation
        x = self.out(x)  # Calculate output
        return x


# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)


# Solstice Deep Q-Learning
class SolsticeDQL:
    learning_rate_a = 0.001  # learning rate (alpha)
    discount_factor_g = 0.9  # discount rate (gamma)
    network_sync_rate = 10  # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000  # size of replay memory
    mini_batch_size = 32  # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()  # NN Loss function. MSE=Mean Squared Error can be swapped to something else.

    optimizer = None  # NN Optimizer. Initialize later.

    # Train the Solstice environment
    def train(self, game: SolsticeGame, episodes):

        epsilon = 1  # 1 = 100% random actions
        memory = ReplayMemory(int(max(self.replay_memory_size, episodes)))  # Size is 1000

        if self.expert_gameplay_enabled and self.expert_gameplay:
            game.RenderScreen("Expert train! =" + str(
                len(self.expert_gameplay)) + " steps" + f"\nTrain in progress  with {game.level_channels_count} ch\n" + f"Train device {device}\n" + str(
                episodes) + " episodes", "evil")
        else:
            game.RenderScreen(
                f"Train in progress with {game.level_channels_count} ch\n" + f"Train device {device}\n" + str(
                    episodes) + " episodes", "evil")

        # Create policy and target network.
        policy_dqn = DQN(in_channels=game.level_channels_count, map_height=game.level_height,
                         map_width=game.level_width, h1_nodes=game.level_size_for_hidden_layer,
                         out_actions=game.action_size).to(device)
        target_dqn = DQN(in_channels=game.level_channels_count, map_height=game.level_height,
                         map_width=game.level_width, h1_nodes=game.level_size_for_hidden_layer,
                         out_actions=game.action_size).to(device)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy network - random, before training:')
        interesting_states = game.generate_multi_channel_state()
        self.print_dqn(policy_dqn, interesting_states)

        # Policy network optimizer. "Adam" optimizer can be swapped to something else.
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count = 0

        expert_episode = 0;
        for i in range(episodes):
            game.SetTitle("Train episode {}/{}".format(i + 1, episodes))

            is_terminated = False  # True when agent dies or reached goal
            is_truncated = False  # True when agent takes more than X actions
            cumulative_reward = 0
            state, state_tensor, info = game.reset()

            current_expert_episode_steps = [];
            if self.expert_gameplay_enabled and self.expert_gameplay:
                current_expert_episode_steps = self.expert_gameplay.pop(0);  # Here is stored first episode steps

            # Agent navigates map until it dies/reaches goal (terminated), or has taken 200 actions (truncated).
            while (not is_terminated and not is_truncated):

                if self.expert_gameplay_enabled and current_expert_episode_steps:
                    # Preload the replay memory with expert gameplay
                    transition = current_expert_episode_steps.pop(0)  # Get and remove the first element
                    state, state_tensor, action, new_state, new_state_tensor, reward, is_terminated = transition

                    print(
                        f"Processing expert transition with reward {reward} - Terminated {is_terminated}. Expert transitions left: {len(current_expert_episode_steps)}")

                else:

                    # Select action based on epsilon-greedy
                    if random.random() < epsilon:
                        # select random action
                        action = game.action_space_sample()  # actions: 0=left,1=down,2=right,3=up
                    else:
                        # select best action
                        with torch.no_grad():
                            dqn_input = self.state_to_dqn_input(state, state_tensor, game.level_channels_count)
                            action_values = policy_dqn(dqn_input)
                            action = action_values.argmax().item()

                    # Execute action
                    new_state, new_state_tensor, reward, is_terminated, is_truncated, _ = game.step(action)

                cumulative_reward += reward  # Update at each step within the while loop

                # print(f"Adding state_tensor with shape {state_tensor.shape} to memory. Level: {game.level_index}")
                # Save experience into memory
                memory.append((state, state_tensor, action, new_state, new_state_tensor, reward, is_terminated))

                # Move to the next state
                state = new_state
                state_tensor = new_state_tensor

                # Increment step counter
                step_count += 1

            # Keep track of the rewards collected per episode.
            if cumulative_reward >= 1:
                rewards_per_episode[i] = cumulative_reward
                print(f"Saving rewards for episode {i} - {cumulative_reward}")

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode) > 0:
                # print("Enought expiriance has been collected for optimization! - Memory "+str(len(memory)))
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = max(epsilon - 1 / episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        # Save policy
        torch.save(policy_dqn.state_dict(), "dql/solstice_dql_" + str(game.level_index) + ".pt")

        game.RenderScreen("Train completed!\n" + str(episodes) + " episodes", "wizard")

        # Create new graph
        # TODO: i need to reset this plt here, so its not showing later more graphs on the same
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x - 100):(x + 1)])
        plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)

        # Save plots
        plt.savefig("dql/solstice_dql_" + str(game.level_index) + ".png")

        plt.show()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        if len(set([s[1].shape for s in mini_batch])) > 1:
            print("Warning: Inconsistent state tensor sizes in mini-batch.")

        # Extract experiences
        state_tensors = torch.stack([s[1] for s in mini_batch])
        actions = torch.tensor([s[2] for s in mini_batch], dtype=torch.int64).view(-1, 1)
        rewards = torch.tensor([s[5] for s in mini_batch], dtype=torch.float32)
        non_final_next_states = torch.stack([s[4] for s in mini_batch if not s[6]])
        non_final_mask = torch.tensor([not s[6] for s in mini_batch], dtype=torch.bool)

        # Determine the device dynamically
        device = next(policy_dqn.parameters()).device
        state_tensors = state_tensors.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        non_final_next_states = non_final_next_states.to(device)
        non_final_mask = non_final_mask.to(device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        current_q_values = policy_dqn(state_tensors).gather(1, actions).squeeze()

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(len(mini_batch), device=device)
        next_state_values[non_final_mask] = target_dqn(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_q_values = (next_state_values * self.discount_factor_g) + rewards

        # Compute Huber loss
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in policy_dqn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    # Optimize policy network
    def optimizeOld(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                # Agent either reached goal (reward=1+) or died (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(
                            self.state_to_dqn_input(new_state, num_states)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states))
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    '''
    OBSOLETE - must replace with multy channel stuff
    Converts an state (int) to a tensor representation.
    For example, the Solstice 4x4 map has 4x4=16 states numbered from 0 to 15. 

    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''

    def state_to_dqn_input(self, state: int, state_tensor, num_states: int) -> torch.Tensor:

        # TODO: this is needed for multiple channels
        return state_tensor.unsqueeze(0);

        # this was single channel stuff

        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # Run the Solstice game with the learned policy
    def test(self, game: SolsticeGame, episodes):
        # Load learned policy
        policy_dqn = DQN(in_channels=game.level_channels_count, map_height=game.level_height,
                         map_width=game.level_width, h1_nodes=game.level_size_for_hidden_layer,
                         out_actions=game.action_size).to(device)
        policy_dqn.load_state_dict(torch.load("dql/solstice_dql_" + str(game.level_index) + ".pt", map_location=device))
        policy_dqn.eval()  # switch model to evaluation mode

        print('Policy (trained):')
        interesting_states = game.generate_multi_channel_state()
        self.print_dqn(policy_dqn, interesting_states)

        for i in range(episodes):
            game.SetTitle("Test episode {}/{}".format(i + 1, episodes))
            state, state_tensor, info = game.reset()  # Initialize
            is_terminated = False  # True when agent falls in hole or reached goal
            is_truncated = False  # True when agent takes more than 200 actions

            # Agent navigates map until it dies (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while (not is_terminated and not is_truncated):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        is_terminated = True  # To ensure the game will also close if the window's close button is clicked
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            is_terminated = True  # Break out of the loop when Escape is pressed
                            break  # Exit the for loop immediately

                    # Exit the while loop if is_terminated has been set to True
                if is_terminated:
                    break

                # Select best action
                with torch.no_grad():
                    dqn_input = self.state_to_dqn_input(state, state_tensor, game.level_channels_count)
                    action_values = policy_dqn(dqn_input)
                    action = action_values.argmax().item()

                # Execute action
                state, state_tensor, reward, is_terminated, is_truncated, _ = game.step(action)

    # expert gameplay data
    expert_gameplay = []
    expert_gameplay_enabled = False
    expert_gameplay_current_episode = 0

    def prepareNewLevel(self):
        memory = ReplayMemory(self.replay_memory_size)  # Size is 1000
        print("Resetting memory with new level " + str(game.level_index));

    def play(self, game: SolsticeGame):

        game.SetTitle("Play the game")
        state, state_tensor, info = game.reset()
        self.prepareNewLevel()

        disable_display_for_training = True
        training_episodes = 512  # Default value

        # Correctly decode the env.desc to get the map layout
        map_layout = game.map_layout

        action_mapping = {
            pygame.K_LEFT: 0,
            pygame.K_DOWN: 1,
            pygame.K_RIGHT: 2,
            pygame.K_UP: 3,
            pygame.K_r: 'reset',
            pygame.K_s: 'skin',
            pygame.K_t: 'train',
            pygame.K_e: 'test',
            pygame.K_x: 'expert',
            pygame.K_n: 'next',
            pygame.K_p: 'prev',
            pygame.K_c: 'channels',
            pygame.K_PLUS: 'plus',
            pygame.K_EQUALS: 'plus',
            pygame.K_MINUS: 'minus',
            pygame.K_h: 'toggle_display',
        }

        is_terminated = False

        while not is_terminated:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_terminated = True
                elif event.type == pygame.KEYDOWN:
                    if event.key in action_mapping:
                        if action_mapping[event.key] == 'reset':
                            # Regenerate the map and reset position
                            game.map_layout = game.generate_solvable_map(8, 8)
                            state, state_tensor, info = game.reset()
                            self.prepareNewLevel();
                            print("Map regenerated and position reset.")
                        elif action_mapping[event.key] == 'skin':
                            game.skin = random.choice(game.skins)
                            game.render()
                            print(f"Skin changed to {game.skin}.")
                        elif action_mapping[event.key] == 'next':
                            print("Next level loading");
                            state, state_tensor, info = game.NextLevel();
                            game.render()
                            self.prepareNewLevel();
                            print(f"Skin changed to {game.skin}.")
                        elif action_mapping[event.key] == 'prev':
                            print("Prev level loading");
                            state, state_tensor, info = game.PrevLevel();
                            game.render()
                            self.prepareNewLevel();
                            print(f"Skin changed to {game.skin}.")
                        elif action_mapping[event.key] == 'channels':
                            game.generate_multi_channel_state(True)
                            print(f"Displaying channels")
                        elif action_mapping[event.key] == 'toggle_display':
                            disable_display_for_training = not disable_display_for_training;
                            print(f"Hidden training changed to {disable_display_for_training}.")
                            game.RenderScreen(f"Hidden training changed to {disable_display_for_training}.", "wizard")
                        elif action_mapping[event.key] == 'train':
                            self.prepareNewLevel();
                            if (disable_display_for_training):
                                game.DisableDisplay()
                            solsticeDQL.train(game, training_episodes)
                            game.EnableDisplay()
                            print(f"Training finished.")
                            game.RenderScreen(f"Training finished!\nReset the level.", "wizard")
                        elif action_mapping[event.key] == 'plus':  # Increase episodes
                            training_episodes *= 2
                            training_episodes = int(training_episodes)
                            print(f"Training episodes set to {training_episodes}.")
                            game.RenderScreen(f"Training episodes set to {training_episodes}.", "wizard")
                        elif action_mapping[event.key] == 'minus':  # Decrease episodes
                            training_episodes /= 2
                            training_episodes = int(max(2, training_episodes))
                            print(f"Training episodes set to {training_episodes}.")
                            game.RenderScreen(f"Training episodes set to {training_episodes}.", "evil")
                        elif action_mapping[event.key] == 'test':
                            self.prepareNewLevel();
                            print(f"Start testing")
                            solsticeDQL.test(game, 1)
                            print(f"Testing the game completed.")
                            game.RenderScreen(f"Test completed.", "wizard")
                        elif action_mapping[event.key] == 'expert':
                            # Toggling the expert session - if i disable it - im just dropping all the data.. this is not the place, where i save something, this is just a toggle.
                            if (self.expert_gameplay_enabled):
                                self.expert_gameplay_enabled = False
                                self.expert_gameplay_current_episode = 0
                                self.expert_gameplay = []
                                print(f"Ending the expert session")
                                game.RenderScreen(f"Disabling expert session", "evil")
                            else:
                                self.expert_gameplay_enabled = True
                                self.expert_gameplay_current_episode = 0
                                # resetting the expert gameplay data
                                self.expert_gameplay = []
                                state, state_tensor, info = game.reset();
                                self.prepareNewLevel();
                                print(f"Starting the expert gameplay session.")
                                game.RenderScreen(f"Starting the expert gameplay\nsession with =4 episodes", "wizard")
                        else:
                            action = action_mapping[event.key]

                            # Execute the chosen action and record the result
                            new_state, new_state_tensor, reward, is_terminated, is_truncated, _ = game.step(action)
                            # Record the expert's action and the game's response
                            if len(self.expert_gameplay) <= self.expert_gameplay_current_episode:
                                self.expert_gameplay.append([])  # Initialize new episode

                            properList = (
                                state, state_tensor, action, new_state, new_state_tensor, reward, is_terminated)

                            print(
                                f"Adding state_tensor with shape {state_tensor.shape} >> {new_state_tensor.shape} to expert_gameplay memory. Level: {game.level_index}")

                            self.expert_gameplay[self.expert_gameplay_current_episode].append(properList)

                            state = new_state
                            state_tensor = new_state_tensor

                            print(
                                f"Action: {['Left', 'Down', 'Right', 'Up'][action]}, New State: {new_state}, Reward: {reward}")
                            if is_terminated:

                                if (self.expert_gameplay_enabled):
                                    self.expert_gameplay_current_episode += 1;
                                    print("Expert training completed. " + str(
                                        self.expert_gameplay_current_episode) + " from 4")

                                if (self.expert_gameplay_enabled):
                                    if (self.expert_gameplay_current_episode >= 4):
                                        print(
                                            f"Training the game with expert gameplay lenght.{len(self.expert_gameplay)}")
                                        # TODO - something about the expert gameplay if i won - maybe start training with expertgameplay data?
                                        if (disable_display_for_training):
                                            game.DisableDisplay()
                                        solsticeDQL.train(game, training_episodes)
                                        # Inside, before starting training we will check, if there is something insdie expert gameplay, and use that as base knowledge data
                                        game.EnableDisplay()

                                        self.expert_gameplay_enabled = False;
                                        self.expert_gameplay_current_episode = 0;
                                        self.expert_gameplay = []
                                        print("Continue ordinary gameplay after expert training")
                                        is_terminated = False
                                        state, state_tensor, info = game.reset();
                                    else:
                                        is_terminated = False
                                        print("Continue expert gameplay")
                                        state, state_tensor, info = game.reset();

                                else:
                                    if reward >= 1:
                                        state, state_tensor, info = game.Won();
                                        is_terminated = False
                                        self.prepareNewLevel();
                                    else:
                                        state, state_tensor, info = game.Lost();
                                        is_terminated = False
                                        self.prepareNewLevel();

    def print_dqn(self, dqn, state_tensor):
        """
        Prints the DQN output for a list of given state tensors.

        Parameters:
        - dqn: The DQN model to use for generating action values.
        - state_tensors: A list of state tensors to evaluate.
        """
        with torch.no_grad():
            action_values = dqn(state_tensor.unsqueeze(0))  # Ensure it has a batch dimension
            q_values = action_values.squeeze(0).tolist()
            best_action_index = action_values.argmax(dim=1).item()
            best_action = ['Left', 'Down', 'Right', 'Up'][best_action_index]
            q_values_formatted = ' '.join(f"{q:+.2f}" for q in q_values)
            print(f"Best action: {best_action}, Q-values: [{q_values_formatted}]")

    # Print DQN: state, best action, q values
    def print_dqn_old(self, dqn):
        # Get number of input nodes
        num_states = dqn.fc1.in_features

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values_formatted = ''
            dqn_input = self.state_to_dqn_input(s, num_states)
            q_values_tensor = dqn(dqn_input)
            q_values_tensor_list = q_values_tensor.tolist()

            for q in q_values_tensor_list:
                q_values_formatted += "{:+.2f}".format(q) + ' '  # Concatenate q values, format to 2 decimals

            q_values_formatted = q_values_formatted.rstrip()  # Remove space at the end

            # Map the best action to L D R U
            argmax = q_values_tensor.argmax()
            best_action = ['Left', 'Down', 'Right', 'Up'][argmax]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the Solstice map.
            print(f'{s:02},{best_action},[{q_values_formatted}]', end=' ')
            if (s + 1) % 4 == 0:
                print()  # Print a newline every 4 states


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    solsticeDQL = SolsticeDQL()
    game = SolsticeGame(level_index=1, device=device)
    solsticeDQL.play(game)
    game.close()
