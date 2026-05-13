

#include <iostream>

#define PUSH_BOTTON_POWER 0
#define PUSH_BOTTON_MAKE  1



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


/* test */
int main()
{
    bool is_power_on = false;
    bool is_power_on_next = false;
    int index_action = -1;
    double res = 0;

    std::cout << "TEST nachi_machine" << std::endl;
    std::cout << "------" << std::endl;

    is_power_on = false;
    index_action = PUSH_BOTTON_POWER;
    res = nachi_machine(is_power_on, index_action, is_power_on_next);
    std::cout << "power: " << is_power_on << " ,action: " << index_action << std::endl;
    std::cout << "power next: " << is_power_on_next << std::endl;
    std::cout << "reward: " << res << std::endl;

    std::cout << "------" << std::endl;

    is_power_on = true;
    index_action = PUSH_BOTTON_POWER;
    res = nachi_machine(is_power_on, index_action, is_power_on_next);
    std::cout << "power: " << is_power_on << " ,action: " << index_action << std::endl;
    std::cout << "power next: " << is_power_on_next << std::endl;
    std::cout << "reward: " << res << std::endl;

    std::cout << "------" << std::endl;

    is_power_on = true;
    index_action = PUSH_BOTTON_MAKE;
    res = nachi_machine(is_power_on, index_action, is_power_on_next);
    std::cout << "power: " << is_power_on << " ,action: " << index_action << std::endl;
    std::cout << "power next: " << is_power_on_next << std::endl;
    std::cout << "reward: " << res << std::endl;

    std::cout << "------" << std::endl;

    is_power_on = false;
    index_action = PUSH_BOTTON_MAKE;
    res = nachi_machine(is_power_on, index_action, is_power_on_next);
    std::cout << "power: " << is_power_on << " ,action: " << index_action << std::endl;
    std::cout << "power next: " << is_power_on_next << std::endl;
    std::cout << "reward: " << res << std::endl;


    return 0;
}


