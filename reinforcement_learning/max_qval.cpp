

#include <iostream>


int calc_maxq(const bool is_power_on, int num_action, int** q_table)
{
    int max_q = -99999;

    for (int i = 0; i < num_action; i++)     // linear search
    {
        max_q = std::max(q_table[is_power_on][i], max_q);
    }


    return max_q;
}



// test for calc_maxq
int main()
{
    int** q_table;
    int num_action = 10;
    bool is_power_on = true;

    q_table = new int*[2];
    for (int i = 0; i < 2; i++)
    {
        q_table[i] = new int[num_action];
    }

    for (int i = 0; i < num_action; i++)
    {
        q_table[is_power_on][i] = -1 * ((i - 5) * (i - 5)) + 10;
    }

    int max = calc_maxq(is_power_on, num_action, q_table);

    std::cout << max << std::endl;  // i=5, 10


    for (i = 0; i < num_action; i++)
    {
        delete[] q_table[i];
    }
    delete[] q_table;


    return 0;
}



