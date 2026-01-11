Usage
1. Sanity Check (Recommended)
Before training, verify your model architecture and data augmentation pipeline are error-free.

Bash

python scripts/sanity_check.py
Check: Open sanity_check_result.png to confirm masks align with rotated images.

2. Training
Run the main training loop. This script includes a Learning Rate Scheduler, CSV logging, and "Best Model" saving.

Bash

python scripts/train.py
Console: Real-time metrics (Dice Score, Loss) will print to the terminal.

Logs: A training_log.csv file will be generated for post-training analysis.

3. Visualization (TensorBoard)
To view real-time loss curves and training graphs:

Bash

tensorboard --logdir runs
Open http://localhost:6006 in your browser.

4. Inference (Prediction)
To generate predictions on the test set with professional green-screen overlays:

Bash

python scripts/predict_test.py
Output: Images will be saved to the test_predictions/ folder.


To Predict: Run scripts/predict_test.py to generate translucent overlays on unseen images using your best saved checkpoint.

To Evaluate: Open training_log.csv to compare Train Loss vs. Validation Loss and identify overfitting (where Val Loss spikes while Train Loss drops).

To Visualize: Check the saved_images/ folder after every epoch to see side-by-side comparisons of the Model's Prediction vs. Ground Truth.