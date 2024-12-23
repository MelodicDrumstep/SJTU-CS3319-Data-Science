You must install graphviz (see instructions at https://graphviz.gitlab.io/download/) for `plot_model` to work.
INFO:logger_main:BEST PLAYER VERSION: 0
INFO:logger_main:====================
INFO:logger_main:EPISODE 1 OF 30
INFO:logger_main:====================
INFO:logger_main:best_player plays as X
INFO:logger_main:--------------
INFO:logger_main:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_main:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_main:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_main:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_main:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_main:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_main:--------------
INFO:logger_mcts:****** BUILDING NEW MCTS TREE FOR AGENT best_player ******
INFO:logger_mcts:***************************
INFO:logger_mcts:****** SIMULATION 1 ******
INFO:logger_mcts:***************************
INFO:logger_mcts:ROOT NODE...000000000000000000000000000000000000000000000000000000000000000000000000000000000000
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:--------------
...
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:--------------
INFO:logger_mcts:------EVALUATING LEAF------
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...


ITERATION NUMBER 1
BEST PLAYER VERSION 0
SELF PLAYING 30 EPISODES...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 110ms/step
/opt/anaconda3/envs/rl-env/lib/python3.10/site-packages/keras/src/models/functional.py:238: UserWarning: The structure of `inputs` doesn't match the expected structure.
Expected: ['main_input']
Received: inputs=Tensor(shape=(1, 2, 6, 7))
  warnings.warn(msg)
INFO:logger_mcts:PREDICTED VALUE FOR 1: 0.000000
INFO:logger_mcts:added node...000000000000000000000000000000000001000000000000000000000000000000000000000000000000...p = 0.142857
INFO:logger_mcts:added node...000000000000000000000000000000000000100000000000000000000000000000000000000000000000...p = 0.142857
INFO:logger_mcts:added node...000000000000000000000000000000000000010000000000000000000000000000000000000000000000...p = 0.142857
INFO:logger_mcts:added node...000000000000000000000000000000000000001000000000000000000000000000000000000000000000...p = 0.142857
INFO:logger_mcts:added node...000000000000000000000000000000000000000100000000000000000000000000000000000000000000...p = 0.142857
INFO:logger_mcts:added node...000000000000000000000000000000000000000010000000000000000000000000000000000000000000...p = 0.142857
INFO:logger_mcts:added node...000000000000000000000000000000000000000001000000000000000000000000000000000000000000...p = 0.142857
INFO:logger_mcts:------DOING BACKFILL------
INFO:logger_mcts:***************************
INFO:logger_mcts:****** SIMULATION 2 ******
INFO:logger_mcts:***************************
INFO:logger_mcts:ROOT NODE...000000000000000000000000000000000000000000000000000000000000000000000000000000000000
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:--------------
INFO:logger_mcts:CURRENT PLAYER...1
...
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['X', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:--------------
INFO:logger_mcts:------EVALUATING LEAF------
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step
INFO:logger_mcts:PREDICTED VALUE FOR -1: -0.035161
INFO:logger_mcts:added node...000000000000000000000000000000000001000000000000000000000000000000000010000000000000...p = 0.138929
INFO:logger_mcts:added node...000000000000000000000000000000000001000000000000000000000000000000000000000000100000...p = 0.145667
INFO:logger_mcts:added node...000000000000000000000000000000000001000000000000000000000000000000000000000000010000...p = 0.140290
INFO:logger_mcts:added node...000000000000000000000000000000000001000000000000000000000000000000000000000000001000...p = 0.143982
INFO:logger_mcts:added node...000000000000000000000000000000000001000000000000000000000000000000000000000000000100...p = 0.145165
INFO:logger_mcts:added node...000000000000000000000000000000000001000000000000000000000000000000000000000000000010...p = 0.143292
INFO:logger_mcts:added node...000000000000000000000000000000000001000000000000000000000000000000000000000000000001...p = 0.142675
INFO:logger_mcts:------DOING BACKFILL------
INFO:logger_mcts:updating edge with value 0.035161 for player 1... N = 1, W = 0.035161, Q = 0.035161
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['X', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:--------------
INFO:logger_mcts:***************************
INFO:logger_mcts:****** SIMULATION 3 ******
INFO:logger_mcts:***************************
INFO:logger_mcts:ROOT NODE...000000000000000000000000000000000000000000000000000000000000000000000000000000000000
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
...
INFO:logger_mcts:['-', '-', '-', '-', '-', '-', '-']
INFO:logger_mcts:['-', '-', 'X', '-', '-', '-', '-']
INFO:logger_mcts:--------------
INFO:logger_mcts:------EVALUATING LEAF------