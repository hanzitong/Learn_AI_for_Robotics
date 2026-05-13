
#include <iostream>
#include <cmath>
#include <vector>


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


int select_optimal_action(bool is_power_on, int num_action, double** Qtable)
{
    std::vector<int> index_actions = calc_q_max_index(is_power_on, num_action, Qtable);

    return index_actions[rand() % index_actions.size()];    // index_actions.size() != 0
}


int main()
{
    bool is_power_on = false;
    int num_action = 10;
    double** Qtable;
    Qtable = new double*[2];
    for (int i = 0; i < 2; i++)
    {
        Qtable[i] = new double[num_action];
        for (int j = 0; j < num_action; j++)
        {
            Qtable[i][j] = 0.;
        }
    }

    /* prepare Qtable */
    for (int i = 0; i < 2; i++)
    {
        double constant = i * 5;
        for (int j = 0; j < num_action; j++)
        {
            Qtable[i][j] = -1. * std::pow((j-5), 2) + constant;
        }
    }


    /* test */
    int res = select_optimal_action(is_power_on, num_action, Qtable); // index: 5
    std::cout << res << std::endl;


    /* free Qtable */
    for (int i = 0; i < 2; i++)
    {
        delete[] Qtable[i];
    }
    delete[] Qtable;


    return 0;
}

