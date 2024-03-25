# NES Solstice Reimagined 🎮

Welcome to the GitHub repository of NES Solstice Reimagined, a thrilling adventure and strategy game inspired by the classic NES title. Dive into a world of mystery, challenges, and clever gameplay, reimagined for today's technology. Utilizing modern game development techniques and artificial intelligence, we bring an old-school favorite into the new era with enhanced features, engaging gameplay mechanics, and deep learning-based AI opponents.

## Game Overview 🌍

NES Solstice Reimagined invites players into the mystical world of Solstice, where magic, mystery, and challenges abound. Navigate through various levels, each presenting unique puzzles, traps, and enemies. Utilize strategy, quick reflexes, and smart planning to overcome obstacles and unravel the secrets of Solstice.

### Features ✨

- **Dynamic Levels:** Explore different levels, each with its unique layout, traps, and secrets. From the dark dungeons to the majestic castles, every level promises a new adventure.
- **Smart AI:** Battle against deep learning-based AI opponents that adapt to your strategies, making each encounter unpredictable and challenging.
- **Strategic Gameplay:** Use potions, bombs, keys, and more to navigate through levels, solve puzzles, and defeat enemies. Strategy is your best weapon.
- **Retro Aesthetics:** Enjoy a nostalgic trip with graphics and music that pay homage to the original NES Solstice, all while incorporating modern touches.
- **Customizable Skins:** Change the look of your game with various skins, from a mystical forest theme to an icy wonderland.

## Technologies Used 🛠️

This project leverages the power of Python, PyTorch, and Pygame to create an immersive game experience. With PyTorch, we've developed a deep learning model (DQN - Deep Q-Network) to power the AI opponents, providing them with the ability to learn and adapt to the player's gameplay style. Pygame is used for game development, bringing together graphics, sound, and game logic seamlessly.

## How to Play 🕹️

1. **Navigation:** Use the arrow keys to move your character through the game world.
2. **Interact:** Discover and use various items like keys and potions to aid in your quest.
3. **Battle:** Outsmart the AI-controlled enemies using strategic movements and item usage.
4. **Solve Puzzles:** Each level is filled with puzzles that must be solved to progress.

## Levels and Challenges 🏰

- **Dungeon Escape:** Start your adventure by escaping the dark and dangerous dungeons filled with traps and secrets.
- **Forest of Mysteries:** Navigate through a dense forest, solving nature's puzzles and battling mythical creatures.
- **Icy Pathways:** Tread carefully on slippery ice, avoiding deadly falls and cold-hearted enemies.
- **Castle Siege:** Make your way through the enemy-filled castle to confront the final boss and unlock the secrets of Solstice.

## Installation and Setup 🚀

Clone this repository to your local machine, ensure you have Python, PyTorch, and Pygame installed, and you're ready to embark on your Solstice adventure.

```bash
git clone https://github.com/MasterAGB/Solstice.git
cd Solstice
pip install -r requirements.txt
python main.py
```

## Level Tiles and Symbols 🗺️

Embark on a journey through meticulously crafted levels, each dotted with unique tiles that challenge your strategy and reflexes:

- **S (🧙️) Start:** Your adventure begins here. The player's starting point.
- **. (⬜) Space:** Free, unobstructed ground to walk on.
- **W (🧱) Wall:** Impassable barriers that test your route-finding skills.
- **H (🕳️) Hole:** Beware of falls that could end your quest prematurely.
- **M (👾) Monster:** Lurk with caution. These creatures move randomly and are lethal upon touch.
- **F (👹) Ferocious Monster:** A formidable foe that, upon defeat with a Bomb, leaves behind a Key.
- **B (💣) Bomb:** A strategic asset against monsters, especially effective against the Ferocious Monster or groups of Monsters.
- **U (🌀) Unstable ground:** Tread lightly, as these tiles transform into Holes after one step.
- **D (🍄) Disorienter:** A tricky obstacle that alters your controls, neutralized by the Potion.
- **P (⚗️) Potion:** The antidote to the Mushroom's effect, safeguarding you from future disorientation.
- **K (🗝️) Key:** The key to unlocking new paths and gates within the level.
- **C (🔒) Closed Gate:** Opens with a Key, marking potential exits or important areas.
- **G (🏁) Goal:** The ultimate destination of each level, ideally placed to challenge the player's journey.
- **J (🔑) Journey Key:** Opens specific doors within the level, adding layers to the exploration.
- **T (📦) Locked Chest:** Holds a Level Door Key but requires a Press Plate to unlock.
- **R (🔘) Press Plate:** Activates to unlock specific Locked Chests, revealing crucial items.
- **L (🔓) Open Door:** Signifies a path has been cleared, leading to new sections of the level.
- **A (🔒) Access Denied Door:** Obstructs progress, awaiting the right Key to be unlocked.
- **Q (💀) Quest Beast:** Defeat to acquire a Level Door Key, essential for navigating the intricacies of Solstice.

## Level Editor 🛠️

- **Entering Editor Mode:** Press `D` to activate the editor mode, transforming the game experience by allowing you to modify the level layout in real-time.
- **Exiting Editor Mode:** Pressing `D` again copies the current level layout to the clipboard, making it easy to share or modify levels outside the game environment.

## Building an Executable 📦

For those looking to distribute or play NES Solstice Reimagined without the need for a Python environment, follow these steps to create an executable file:

1. Ensure you have PyInstaller installed:
   ```bash
   pip install pyinstaller
   ```
2. Navigate to your project directory and run PyInstaller with your script:
   ```bash
   pyinstaller --onefile --windowed --icon=icon.ico main.py
   ```
3. Find your executable in the `dist` folder.

This process compiles your game into a standalone `.exe` file, incorporating all necessary dependencies, making it easy to share and enjoy the game across Windows systems without additional setup.

## Contributing 🤝

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. Check out our contribution guidelines for more information.

## License 📜

Distributed under the MIT License. See `LICENSE` for more information.

## Final Thoughts 💭

This project is a tribute to the classic NES Solstice game, reimagined with modern technologies and AI. Whether you're a fan of the original or new to the world of Solstice, we hope this game brings you joy, challenge, and nostalgia. Happy gaming! 🎉
