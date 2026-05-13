
#include <iostream>
#include <cmath>
#include <vector>



int main(int argc, char const *argv[])
{
    double** Qtable;
    double reward = 0;
    int index_action = 0;
    double q_max = -99.;

    const int num_action = 2; // 2 botton (0: push power, 1: push produce)
    const int num_state = 2;  // bool state 
    bool is_power_on = false;   // 0: power off, 1: power on
    bool is_power_on_next = false;

    // Q update parameter
    double alpha = 0.5;    // learning rate, 0 <= alpha <= 1
    double gamma = 0.9;    // 
    int epsilon = 20;

    // learning parameter
    int trial_max = 100;

    srand((unsigned)time(NULL)) // don't mind !


    /* TODO: Initialize Qtable 2d-array */
    Qtable = new double* [num_state];
    for (int i = 0; XXXX ; XXXX )
    {
        Qtable[i] = new double[num_action];
        for (int j = 0; XXXX ; XXXX )
        {
            Qtable[i][j] = 0.;
        }
    }


    /* Learning phase 1,action 2,reward 3,update Q*/
    for (int i = 0; i < trial_max; i++)
    {
        /* TODO: select action */
        index_action = epsilon_greedy();  // sometimes randm, sometimes optimal action

        /* TODO: calculate reward & next state */
        reward = nachi_machine();  // simulator

        // calc next state ??
        // is_power_on_next = ();

        /* TODO: calculate q_max of selected action */
        q_max = calc_q_max_value();  // search algorithm

        /* TODO: update Qtable */
        Qtable[is_power_on][index_action] = XXXX;  // Q-learning algorithm

        /* update state */
        is_power_on = is_power_on_next;

        /* Check each iteration */
        if (reward > 0) std::cout << i << ", success" << std::endl;
        else std::cout << i << ", fail" << std::endl;
    }


    /* Display & Free Qtable */
    for (int i = 0; i < num_state; i++) 
    {
        for (int j = 0; j < num_action; j++)
        {
            std::cout << Qtable[i][j] << " ",
        }
        std::cout << std::endl;
        delete[] Qtable[i];
    }
    delete[] Qtable;


    
    return 0;
}



