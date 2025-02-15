import numpy as np
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    """ MCTS Node of Tree.
    Q : its own value
    P : prior probability
    u : visit-count-adjusted prior score
    """
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {} 
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors, forbidden_moves, is_you_black):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability according to the policy function.
        """
        for action, prob in action_priors:
            if is_you_black and action in forbidden_moves : continue
            if action not in self._children : self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors."""
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class VanillaMCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000): #c_puct=5, n_playout=10000
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        node = self._root
        while(1):
            if node.is_leaf(): 
                break
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, leaf_value = self._policy(state)
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs, state.forbidden_moves, state.is_you_black())
        else:
            leaf_value = 0.0 if winner == -1 else (1.0 if winner == state.get_current_player() else -1.0)
        node.update_recursive(-leaf_value)


    def get_move_probs(self, state, temp=1e-3):
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        total_visits = sum(v for _, v in act_visits)

        restricted_regions = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
                            (1, 0), (1, 1), (1, 2), (1, 3), (1, 5), (1, 6), (1, 7), (1, 8),
                            (2, 0), (2, 8), (8, 3), (8, 4), (8, 5)]
        restricted_visits = sum(v for m, v in act_visits if tuple(state.move_to_location(m)) in restricted_regions)

        restricted_prob = restricted_visits / total_visits if total_visits > 0 else 0
        
        print(f"Restricted region visit probability: {restricted_prob:.4f}")

        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        if last_move in self._root._children:
            self._root = self._root._children[last_move] 
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class VanillaMCTSPlayer(object):
    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = VanillaMCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        move_probs = np.zeros(board.width*board.height)
        if board.width*board.height - len(board.states) > 0:
            acts, probs = self.mcts.get_move_probs(board, temp)      
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                move = np.random.choice(acts, p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(acts, p=probs) 
                self.mcts.update_with_move(-1)

            h, w = board.move_to_location(move)
            print(f"Position of Vanilla MCTS is : ({h}, {w})")

            if return_prob : return move, move_probs
            else : return move
        
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)