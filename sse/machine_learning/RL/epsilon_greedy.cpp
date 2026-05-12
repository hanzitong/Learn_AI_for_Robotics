

#include <iostream>


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

    return 0;
}
