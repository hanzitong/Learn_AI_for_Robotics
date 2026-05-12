
#include <iostream>
#include <cmath>


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
    

    return index_action;
}



int main(int argc, char const *argv[])
{
    bool is_power_on = false;
    int num_action = 10;
    // double** Qtable = new double[2][10];
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


    for (int i = 0; i < 2; i++)
    {
        double constant = i * 5;
        for (int j = 0; j < num_action; j++)
        {
            Qtable[i][j] = -1. * std::pow((j-5), 2) + constant;
        }
    }

    int res = select_action(is_power_on, num_action, Qtable); // j = 5, 0
    std::cout << res << std::endl;


    for (int i = 0; i < 2; i++)
    {
        delete[] Qtable[i];
    }
    delete[] Qtable;


    return 0;
}

