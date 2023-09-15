# Reinforcement Learning Project - Predicting the buying behavior of a hydroelectric dam

We built an environment representing a hydroelectric dam for water storage. With this application, we aimed to find the best time-points to buy or sell water to maximise the revenue. We implemented three algorithm, one baseline "buy low sell high" algorithm, tabular Q-Learning and Double Deep Q-Learning. The main file runs the quantiles baseline algorithm, since it had the highest validation return. The implementation of the Double-Deep Q-Learning and the Tabular Q-Learning can be found in the files run_ddq.py and run_tab_q.py. A complete report of the project can be found in the report.pdf file.

## Running the Main file
To run the main file call the following prompt with the file path.

```{objectives}
python main.py -P PATH
```
For example,

```{objectives}
python main.py -P validate.xlsx
```

