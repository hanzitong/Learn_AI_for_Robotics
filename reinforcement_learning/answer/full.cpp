
#include <iostream>
#include <ctime>
#include <vector>
#include <algorithm>


int epsilon_greedy(int epsilon, bool is_power_on, int num_action, double** Qtable);
int select_optimal_action(bool is_power_on, int num_action, double** Qtable);
std::vector<int> calc_q_max_index(bool is_power_on, int num_action, double** Qtable);
double calc_q_max(bool is_power_on_next, int num_action, double** Qtable);
double nachi_machine(bool is_power_on, int index_action, bool& is_power_on_next);


int epsilon_greedy(int epsilon, bool is_power_on, int num_action, double** Qtable)
{
    int index_action = -99;

    if (epsilon > rand() % 100)
    {
        // choose random action
        std::cout << "random action: ";
        index_action = rand() % num_action;
    }
    else
    {
        std::cout << "q-matrix based action: ";
        index_action = select_optimal_action(is_power_on, num_action, Qtable);
    }


    return index_action;
}


int select_optimal_action(bool is_power_on, int num_action, double** Qtable)
{
    std::vector<int> index_actions = calc_q_max_index(is_power_on, num_action, Qtable);

    return index_actions[rand() % index_actions.size()];    // index_actions.size() != 0
}


std::vector<int> calc_q_max_index(bool is_power_on, int num_action, double** Qtable)
{
    double q_max = Qtable[is_power_on][0];
    std::vector<int> index_actions;
    index_actions.push_back(0);

    for (int i = 1; i < num_action; i++)
    {
        if (q_max < Qtable[is_power_on][i])
        {
            q_max = Qtable[is_power_on][i];
            index_actions.clear();
            index_actions.push_back(i);
        }
        else if (q_max == Qtable[is_power_on][i])
        {
            index_actions.push_back(i);
        }
    }

    return index_actions;
}


double calc_q_max_value(bool is_power_on, int num_action, double** Qtable)
{
    double q_max = Qtable[is_power_on][0];

    for (int i = 1; i < num_action; i++)
    {
        q_max = std::max(q_max, Qtable[is_power_on][i]);
    }

    return q_max;
}


// memo: bool&(reference) is not good (specify input or output)
double nachi_machine(bool is_power_on, int index_action, bool& is_power_on_next)
{
    int reward = 0;

    if (index_action == 0)  // action 0: push power botton
    {
        is_power_on_next = !is_power_on;
        reward = 0;
    }
    else    // action 1: push produce botton
    {
        is_power_on_next = is_power_on;

        if (is_power_on) reward = 10;
        else reward = 0;
    }

    return reward;
}
 

int main()
{
    double ** Qtable;
    double q_max = -99.;
    double reward = 0.;
    double alpha = 0.5;
    double gamma = 0.9;
    int epsilon = 20;
    int trial_max = 100;
    int num_action =2;
    int num_state = 2;
    int index_action = 0;
    bool is_power_on = false;
    bool is_power_on_next = false;

    srand((unsigned)time(NULL));

    Qtable = new double*[num_state];
    for (int i = 0; i < num_state; i++)
    {
        Qtable[i] = new double[num_action];
        for (int j = 0; j < num_action; j++)
        {
            Qtable[i][j] = 0.;
            // Qtable[i][j] = rand() % 10;
        }
    }


    std::vector<int> index_actions(num_action, 0);

    for (int i = 0; i < trial_max; i++)
    {
        index_action = epsilon_greedy(epsilon, is_power_on, num_action, Qtable);
        reward = nachi_machine(is_power_on, index_action, is_power_on_next);
        q_max = calc_q_max_value(is_power_on_next, num_action, Qtable);
        Qtable[is_power_on][index_action] = (1. - alpha) * Qtable[is_power_on][index_action] + alpha * (reward + gamma * q_max);
        is_power_on = is_power_on_next;

        if (reward > 0) std::cout << i << ", success" << std::endl;
        else std::cout << i << ", fail" << std::endl;
    }

    // display and free Qtable
    std::cout << "=== Qtable ===" << std::endl;
    for (int i = 0; i < num_state; i++)
    {
        for (int j = 0; j < num_action; j++)
        {
            std::cout << Qtable[i][j] << " ";
        }
        std::cout << std::endl;
        delete[] Qtable[i];
    }
    delete[] Qtable;


    return 0;
}


