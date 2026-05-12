

#include <iostream>
#include <cmath>


#define PUSH_BOTTON_POWER 0
#define PUSH_BOTTON_MAKE  1


double nachi_machine(bool is_power_on, bool& is_power_on_next, const int action)    // simulator
{
    double reward = -1;

    if (action == PUSH_BOTTON_POWER)
    {
        is_power_on_next = !is_power_on;
        reward = 0;
    }
    else if (action == PUSH_BOTTON_MAKE)
    {
        if (is_power_on)
        {
            reward = 10;
        } else {
            reward = 0;
        }
    } else {
        std::cout << "Error: nachi_machine()" << std::endl;
    }

    return reward;
}


double calc_maxq(const bool is_power_on, int num_action, double** q_table)
{
    double max_q = -99999;

    for (int i = 0; i < num_action; i++)     // linear search
    {
        max_q = std::max(q_table[is_power_on][i], max_q);
    }


    return max_q;
}


int select_action(bool is_power_on, int num_action, double**Qtable)
{
    double max_val = Qtable[is_power_on][0];
    int num_max = 1;    // init value = 1

    int* index_max = new int[num_action];
    index_max[0] = 0;   // init value

    int index_action;   // action to return


    for (int i = 1; i<num_action; ++i)
    {
        if (Qtable[is_power_on][i] > max_val)
        {
            max_val = Qtable[is_power_on][i];   // update 
            num_max = 1;    // init
            index_max[0] = i;   // init
        }
        else if (Qtable[is_power_on][i] == max_val)
        {
            num_max++;
            index_max[num_max - 1] = i; // push back
        }
    }

    index_action = index_max[rand() % num_max];
    delete[] index_max;
    

    return index_action;
}



int epsilon_greedy(int epsilon, bool is_power_on, int num_action, double** Qtable)
{
    int index_action;

    if (epsilon > rand() % 100)
    {
        index_action = rand() % num_action;
        std::cout << "random action" << std::endl;
    }
    else {
        index_action = select_action(is_power_on, num_action, Qtable);
        std::cout << "max Qvalue action" << std::endl;
    }


    return index_action;
}


int main()
{
    int num_action = 2;
    // int num_state = 2;
    int index_action = 0;
    double reward = 0;
    bool is_power_on = false;
    bool is_power_on_next = false;

    double Q_max = 0.;
    double alpha = 0.5;
    double gamma = 0.9;
    int epsilon = 50;
    int trial_max = 100;
    double** Qtable = new double*[2];
    for (int i = 0; i < 2; i++)
    {
        Qtable[i] = new double[num_action];
        for (int j = 0; j < num_action; j++)
        {
            Qtable[i][j] = 0;
        }
    }


    srand((unsigned)time(NULL));


    for (int i = 0; i < 100; i++)
    {
        // select action
        index_action = epsilon_greedy(epsilon, is_power_on, num_action, Qtable);

        // calc reward
        reward = nachi_machine(is_power_on, is_power_on_next, index_action);

        // calc q_max
        Q_max = calc_maxq(is_power_on_next, num_action, Qtable);

        // update Qtable
        Qtable[is_power_on][index_action] = (1 - alpha) * Qtable[is_power_on][index_action] + alpha * (reward + gamma * Q_max);

        // update state
        is_power_on = is_power_on_next;


        // display results
        std::cout << i << std::endl;
        if (reward >= 0) std::cout << "success" << std::endl;
    }

    // display Qtable
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < num_action; j++)
        {
            std::cout << Qtable[i][j] << " ";
        }
        std::cout << std::endl;
    }


    // free Qtable
    for (int i = 0; i < 2; i++)
    {
        delete[] Qtable[i];
    }
    delete[] Qtable;



    return 0;
}



