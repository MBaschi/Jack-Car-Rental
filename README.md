In the repo i work on exercise 4.2 of the book 
Reinforcement Learning:
An Introduction
second edition
Richard S. Sutton and Andrew G. Bart

Text of the exercise: 
Jack manages two locations for a nationwide car
rental company. Each day, some number of customers arrive at each location to rent cars.
If Jack has a car available, he rents it out and is credited $10 by the national company.
If he is out of cars at that location, then the business is lost. Cars become available for
renting the day after they are returned. To help ensure that cars are available where
they are needed, Jack can move them between the two locations overnight, at a cost of
$2 per car moved. We assume that the number of cars requested and returned at each
location are Poisson random variables, meaning that the probability that the number is
n is  (poisson dist formula with parameter lambda). Suppose lambda is 3 and 4 for rental requests at
the first and second locations and 3 and 2 for returns. To simplify the problem slightly,
we assume that there can be no more than 20 cars at each location (any additional cars
are returned to the nationwide company, and thus disappear from the problem) and a
maximum of five cars can be moved from one location to the other in one night. We take
the discount rate to be 0.9  and formulate this as a continuing finite MDP, where
the time steps are days, the state is the number of cars at each location at the end of
the day, and the actions are the net numbers of cars moved between the two locations
overnight. Figure 4.2 shows the sequence of policies found by policy iteration starting
from the policy that never moves any cars.

Altough the object can be solved with motecarlo or TD learning i sticked as much as
possible to the dynamic programming approach. This require all transition probabilities
to be known (and precompute to avoid repeting the calculation and taking foreverer)

