import os
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(dir_path, "system_stransition_probability")
trasition_probability_filenames = os.path.join(save_path, "prob_values.pkl")

MAX_CAR = 20
# poisson parameter of the problem
u_rent_A = 3
u_rent_B = 4
u_return_A = 3
u_return_B = 2

# construct all possible state: each location can have maximum 20 cars
possible_states = []
for i in range(MAX_CAR + 1):
    for j in range(MAX_CAR + 1):
        possible_states.append([i, j])
possible_states = np.array(possible_states)

# construct action space: i can move maximum 5 cars (> 0 form A to B, < 0 from B to A)
possible_action = range(-5, 6)

# the min reward is the case in wich i move the maximum ammount of car and gain nothing (2$ * 5 car moved)
# The best possible case in the one in wich i move no car and all the car are rented (10$ * 40 car)
# the reward is always a multiple of 5
possible_rewards = range(-2 * 5, 10 * 40 + 1, 2)
# to save computation time i will consider the poisson distribution of probability of renting car
# the value of the PMF of a poisson dist with lambda =4 have negligible value above 7 so i will the consider the
# max possible renting value as 14
possible_rewards = range(-2 * 5, 10 * 14 + 1, 2)


class TruncatedPoissonPmf:
    """Object containing precomputed value of a truncated poisson distribution probability mass function
    The class is used to save time by precomputing (and saving in this class) the values
    """

    def __init__(self, lambda_value, possible_n, possible_n_max) -> None:
        self.lambda_value = lambda_value
        self.values = self.precompute_values(possible_n, possible_n_max)

    def precompute_values(self, possible_n, possible_n_max):
        values = {}
        for n in possible_n:
            for n_max in possible_n_max:
                values[(n, n_max)] = self.truncated_poisson_pmf(
                    n, self.lambda_value, n_max
                )
        return values

    @staticmethod
    def truncated_poisson_pmf(k, lambda_value, max_k):
        if k > max_k:
            return 0
        normalization_constant = sum(
            stats.poisson.pmf(i, lambda_value) for i in range(max_k + 1)
        )
        return stats.poisson.pmf(k, lambda_value) / normalization_constant

    def __call__(self, n, n_max) -> float:
        return self.values[(n, n_max)]


class PoissonPmf:
    """Object containing precomputed value of a poisson distribution probability mass function
    The class is used to save time by precomputing (and saving in this class) the values
    """

    def __init__(self, lambda_value, possible_n) -> None:
        self.lambda_value = lambda_value
        self.values = self.precompute_values(possible_n)

    def precompute_values(self, possible_n):
        values = {}
        for n in possible_n:
            values[n] = stats.poisson.pmf(n, self.lambda_value)
        return values

    def __call__(self, n) -> float:
        return self.values[n]


# precompute probabiliyt of renting car in A and B and returning M car
truncated_poisson_pmf_rent_A = TruncatedPoissonPmf(
    lambda_value=u_rent_A, possible_n=range(29), possible_n_max=range(29)
)
truncated_poisson_pmf_rent_B = TruncatedPoissonPmf(
    lambda_value=u_rent_B, possible_n=range(29), possible_n_max=range(29)
)
poisson_pmf_of_returning_m_car = PoissonPmf(
    lambda_value=u_return_A + u_return_B, possible_n=range(29)
)


class SystemTransitionProbabilities:
    """I precompute the system transition probbilities considering all states and rewards"""

    def __init__(self, filename) -> None:
        if os.path.exists(filename):
            self.values = self.load_from_file(filename)

        else:
            self.values = self.precompute_values()
            self.save_to_file(filename)

    def precompute_values(self):
        values = {}
        print("Precomputing system transition probabilities...")
        for state in tqdm(possible_states):
            for future_state in possible_states:
                for action in possible_action:
                    if self.transition_is_possible(future_state, state, action):
                        for reward in possible_rewards:
                            if self.transition_is_possible(
                                future_state, state, action, reward
                            ):

                                values[
                                    (tuple(future_state), reward, tuple(state), action)
                                ] = self.transition_probailities(
                                    future_state, reward, state, action
                                )

        return values

    def transition_probailities(self, future_state, reward, state, action):
        """Proability of moving from to a certain state to another with a certain reward choosed a certain action.
        This contain the dynamics and description of the system"""
        p = 0
        # at the end of a day J have car_i number of car in location i
        car_A = state[0]
        car_B = state[1]

        # at the beginning of the next day J have car_i_available number of car in location i
        car_A_available = car_A + action
        car_B_available = car_B - action

        # actually this control is also before (inside the policy_evaluation()) but i inserted here for readability and conceptual understanding
        if car_A_available < 0 or car_B_available < 0:
            return 0

        # J return to the national company car that can't keep
        if car_A_available > 20:
            car_A_available = 20
        if car_B_available > 20:
            car_B_available = 20

        # at the end of the next day J have car_i_future number of car in location i
        car_A_future = future_state[0]
        car_B_future = future_state[1]

        # the total car rentend and returned is known
        total_car_rented = (
            reward + 2 * abs(action)
        ) / 10  # J gain 10 for each car rented and lose 2 for each car moved
        total_car_returned = (
            (car_A_future + car_B_future) + total_car_rented - (car_A + car_B)
        )  # the returned car cover the difference in the total number of car + the rented one

        if total_car_rented < 0 or total_car_returned < 0:
            # impossible transition
            return 0

        N = total_car_rented
        M = total_car_returned

        # PROBABILITY OF RENTING N CAR
        # the probability have to take into account all the possible combination of renting
        probability_of_renting_N_car = 0

        for n_A in range(int(min(N, car_A_available) + 1)):
            # cycle trough all possible rented car in A

            n_B = (
                N - n_A
            )  # this is the number of car that have to be rented in B in order to have exactly N car rented
            if n_B <= car_B_available:
                prob_A = truncated_poisson_pmf_rent_A(n=n_A, n_max=car_A_available)
                prob_B = truncated_poisson_pmf_rent_B(n=n_B, n_max=car_B_available)
                probability_of_renting_N_car += prob_A * prob_B

        # PROBABILITY OF RETURNING M CAR
        # since the probabilities on the two sites is a poisson distribution also the the overall propability is a poisson
        probability_of_returning_M_car = poisson_pmf_of_returning_m_car(n=M)

        # jointed probability of renting N car and returning M car (transition probability)
        p = probability_of_renting_N_car * probability_of_returning_M_car
        return p

    @staticmethod
    def transition_is_possible(future_state, state, action, reward=None):
        """In order to save computational time the transition with low probabilities
        (in wich too many car have to be rented and returned in on place) are not considered.
        Consider that the value of PMF of a poisson distribution with lambda = 4 (biggest value given)
        is 0.05 at 7, so all transition with a variation bigger than 7 car (14 in total ) are really umprobable
        """
        # at the end of a day J have car_i number of car in location i
        car_A = state[0]
        car_B = state[1]

        car_A_available = car_A + action
        car_B_available = car_B - action

        # move more car than available
        car_A_future = future_state[0]
        car_B_future = future_state[1]

        # difference between initial and final state too big
        if (
            np.abs(car_A_future - car_A_available) > 7
            or np.abs(car_B_future - car_B_available) > 7
        ):
            return False

        # reward impossible or to umprobable to achieve
        if reward is not None:
            # reward = car_rented*10 - car_moved*2 -> car_rented = (reward + car_moved*2)/10
            total_car_rented = (reward + 2 * abs(action)) / 10
            if total_car_rented % 1 != 0:
                # if the number is not integer it means that is an impossible reward with
                # this action
                return False

            total_car_variation = (car_A_future + car_B_future) - (car_A + car_B)
            total_car_returned = total_car_variation + total_car_rented

            if total_car_returned or total_car_rented < 0:
                return False

            # total_car_rented or returned too big or negative
            if 14 < total_car_returned < 0 or total_car_rented > 14:
                return False

        return True

    def __call__(self, future_state, reward, state, action) -> float:
        return self.values[(tuple(future_state), reward, tuple(state), action)]

    def save_to_file(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.values, f)
        print("Transtion probabilities values saved at " + filename)

    def load_from_file(self, filename):
        with open(filename, "rb") as f:
            values = pickle.load(f)
        print("Transtion probabilities values loaded from " + filename)
        return values


transition_probabilities = SystemTransitionProbabilities(
    filename=trasition_probability_filenames
)

# Free some RAM memory by deleting the Poisson distribution tables
# No longer needed after the transition probabilities are calculated
del truncated_poisson_pmf_rent_A
del truncated_poisson_pmf_rent_B
del poisson_pmf_of_returning_m_car


class Policy:
    """Given a state the policy return the action to take"""

    def __init__(self):
        self.policy = np.random.randint(-5, 6, (MAX_CAR + 1, MAX_CAR + 1))

    def __setitem__(self, state, action):
        self.policy[tuple(state)] = action

    def __call__(self, state):
        return self.policy[tuple(state)]

    def show_or_save_colormap(self, filename):
        """ "Plot (or save if the filename is provided) the colormap of the policy"""
        # Create a color map
        plt.imshow(self.policy, cmap="viridis")
        plt.colorbar()  # Add a color bar to show the scale
        plt.title("Policy ")
        plt.xlabel("Car A")
        plt.ylabel("Car B")
        plt.gca().invert_yaxis()

        if filename:
            plt.savefig(filename)  # Save the figure to a file
        else:
            plt.show()  # Display the figure
        plt.close()  # Close the figure to free up memory


policy = Policy()


# V
class SatateValueFunction:
    """Given a state the policy return the value of that state (that correspond to the final total reward)"""

    def __init__(self):
        self.value = np.zeros((MAX_CAR + 1, MAX_CAR + 1))

    def __setitem__(self, state, value):
        self.value[tuple(state)] = value

    def __call__(self, state):
        return self.value[tuple(state)]

    def show_colormap(self):
        # Create a color map
        plt.imshow(self.value, cmap="viridis", interpolation="nearest")
        plt.colorbar()  # Add a color bar to show the scale
        plt.title("State value function")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()


state_value_function = SatateValueFunction()


# q
def action_value_function(state, action, gamma):
    """The action value function return the expected final reward of taking an action in a state.
    To compute it is needed the action, the state, the transition probabilities and the discount factor
    It is also supposed that all possible states and rewards are known"""
    q = 0
    for future_state in possible_states:
        # to save some computation time future states that are too far from the consideed state
        # are not considered since they won't contribute to the action value function
        if transition_probabilities.transition_is_possible(future_state, state, action):

            for reward in possible_rewards:
                if transition_probabilities.transition_is_possible(
                    future_state, state, action, reward
                ):
                    q += transition_probabilities(
                        future_state, reward, state, action
                    ) * (reward + gamma * state_value_function(future_state))

    return q


def policy_evaluation():

    while True:
        max_var = -1
        for state in possible_states:

            old_value = state_value_function(state)
            action = policy(state)

            if state[0] + action < 0 or state[1] - action < 0:
                # if we more more car than available the value is negative and the simulation end
                new_value = -2 * action
            else:
                new_value = action_value_function(state, action, gamma=0.9)

            state_value_function[state] = new_value
            max_var = max(max_var, abs(old_value - new_value))

        print(f"max value function correction: {max_var}")
        # state_value_function.show_colormap()

        if max_var < 1e-3:
            break


def argmax_of_action_value_function(state):
    """Return the action that maximizes the action value function"""

    max_val = None
    best_action = None
    for action in possible_action:
        action_val = action_value_function(state, action, gamma=0.9)
        if max_val is None:
            max_val = action_val

        if max_val is not None and action_val > max_val:
            best_action = action
            max_val = action_val

    if best_action is None:
        return 0

    return best_action


def policy_improvement():
    all_action_unchanged = True
    for state in tqdm(possible_states):
        old_action = policy(state)
        new_action = argmax_of_action_value_function(state)
        policy[state] = new_action
        if old_action != new_action:
            all_action_unchanged = False

    return all_action_unchanged


def find_optimal_policy():
    # initialization
    policy_unstable = True
    iteration = 0
    while policy_unstable:
        # policy evaluation
        print(f"Evaluating the policy {iteration}")
        policy_evaluation()
        # policy improvment
        print("Improving the policy")
        all_action_unchanged = policy_improvement()

        # Save the colormap image

        filename = os.path.join(dir_path, f"policy_iteration_{iteration}.png")

        policy.show_or_save_colormap(filename)
        iteration += 1
        if all_action_unchanged:
            policy_unstable = False


if __name__ == "__main__":
    find_optimal_policy()
