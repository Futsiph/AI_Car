# AI_Car
Autonomous car development using Artificial Intelligence.

## Work in Progress (WIP)
- **Data Preprocessing**: Implementing scripts to process videos and images for model training.

## Future Work
- **Model Training**: Develop and train the neural network for autonomous driving.
- **Testing**: Validate the model using test datasets and real-world simulations.
- **Optimization**: Improve driving logic and performance.

## History

### 2026-06-08
- **Project Initialization**: Created the basic project structure.
- **Environment Setup**: Configured the project and added `.gitignore` to exclude large data files and IDE-specific configurations.
- **Data Collection**: Gathered initial video footage and extracted frames for training.

### 2026-06-09
- **Data Augmentation**: Implementing techniques to expand the dataset using existing resources.

### 2026-06-10
- **Path Detection**: Investigating automated path detection methods; planning manual labeling if automation proves insufficient.

### 2026-06-11
- **Path Detection**: The Canny method doesn't work because of light reflections. I need to try recording manual driving with an input logger and then retry the Canny method on that data. If it doesn't work, I will need to manually label every frame.

### 2026-06-12
- **Learn**: I learned some basics of mechatronics and read the Arduino documentation.