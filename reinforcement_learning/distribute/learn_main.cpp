
#include <iostream>
#include <cmath>
#include <vector>



int main(int argc, char const *argv[])
{
    double** Qtable;
    const int num_action = 2; // 2 botton (0: push power, 1: push produce)
    const int num_state = 2;  // bool state 
    bool is_power_on = false;   // 0: power off, 1: power on
    double reward = 0;

    // Q-learning parameter
    double alpha = ;    // learning rate, 0 <= alpha <= 1
    double gamma = 0.8; // 


    for (int i = 0; i < 100; i++)
    {
        // calc_reward
        // reward = nachi_machine();

        // select_action

        // update_q

        // update_state


    }
    
    return 0;
}



