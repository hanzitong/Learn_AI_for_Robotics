

#include <iostream>



double calc_q_max_value(bool is_power_on, int num_action, double** Qtable)
{
    double q_max = Qtable[is_power_on][0];

    for (int i = 1; i < num_action; i++)
    {
        q_max = std::max(q_max, Qtable[is_power_on][i]);
    }

    return q_max;
}


/* test */
int main()
{
    double** q_table;
    int num_action = 10;
    bool is_power_on = true;

    q_table = new double*[2];
    for (int i = 0; i < 2; i++)
    {
        q_table[i] = new double[num_action];
    }

    for (int i = 0; i < num_action; i++)
    {
        q_table[is_power_on][i] = -1 * ((i - 5) * (i - 5)) + 10;
    }

    int q_max = calc_q_max_value(is_power_on, num_action, q_table);

    std::cout << q_max << std::endl;  // i=5, 10


    for (int i = 0; i < is_power_on; i++)
    {
        delete[] q_table[i];
    }
    delete[] q_table;


    return 0;
}



