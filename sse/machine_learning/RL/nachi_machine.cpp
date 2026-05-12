

#include <iostream>

#define PUSH_BOTTON_POWER 0
#define PUSH_BOTTON_MAKE  1


double nachi_machine(bool& is_power_on, const int action)    // simulator
{
    double reward = -1;

    if (action == PUSH_BOTTON_POWER)
    {
        is_power_on = !is_power_on;
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


void update_q(const bool is_power_on, const int action, int& q_table)  // eplilon_greedy
{
    double res = (1 - alpha) * Qtable[i][j] + alpha * (reward + gamma * Q_max);
    Qtable[i][j] = res;

    return res;
}


void select_action_vacant()
{

    return;
}


int main()
{
    bool is_power_on = false;
    int action = -1;
    double res = 0;

    std::cout << "TEST nachi_machine" << std::endl;
    std::cout << "------" << std::endl;

    is_power_on = false;
    action = PUSH_BOTTON_POWER;
    res = nachi_machine(is_power_on, action);
    std::cout << "power: " << is_power_on << " ,action: " << action << std::endl;
    std::cout << "reward: " << res << std::endl;

    std::cout << "------" << std::endl;

    is_power_on = true;
    action = PUSH_BOTTON_POWER;
    res = nachi_machine(is_power_on, action);
    std::cout << "power: " << is_power_on << " ,action: " << action << std::endl;
    std::cout << "reward: " << res << std::endl;

    std::cout << "------" << std::endl;

    is_power_on = true;
    action = PUSH_BOTTON_MAKE;
    res = nachi_machine(is_power_on, action);
    std::cout << "power: " << is_power_on << " ,action: " << action << std::endl;
    std::cout << "reward: " << res << std::endl;

    std::cout << "------" << std::endl;

    is_power_on = false;
    action = PUSH_BOTTON_MAKE;
    res = nachi_machine(is_power_on, action);
    std::cout << "power: " << is_power_on << " ,action: " << action << std::endl;
    std::cout << "reward: " << res << std::endl;


    return 0;
}


