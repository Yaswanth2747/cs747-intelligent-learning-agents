#include <random>
#include <iostream>
#include <stdio.h>
#include <iomanip>
using namespace std;

/* 
This code is to implement the epsilon greedy algorithms, a few variants that were 
taught by Prof Shivram Kalyankrishnan in the Autumn Semester 2025 for the course CS747
--------------------------------------------------------------------------------------
e Greedy - 1 Algorithm:
--------------------------------------------------------------------------------------
    Suppose there is an agent that has a horizon access of T steps, now the agent has 
    to choose an action (arm) at each step. For simplicity say that each of the arm is
    a bernoulli arm, i.e., it has a reward of 1 with probability p and 0 with 
    probability (1-p). Now the agent has the dilemma of exploration vs exploitation.
    Now in e Greedy - 1, the agent will first explore in the first eT steps, that 
    exploration can be either deterministic or random, deterministic in the sense that
    we explore all arms equal number of times say in a round robin fashion, or in a random
    manner, i.e., we choose an arm randomly. After the exploration phase, the agent will
    exploit the best arm, i.e., the arm with the highest empirical mean, for the
    remaining (1-e)T steps.
--------------------------------------------------------------------------------------
e Greedy - 2 Algorithm:
--------------------------------------------------------------------------------------
    Suppose there is an agent that has a horizon access of T steps, now the agent has 
    to choose an action (arm) at each step. For simplicity say that each of the arm is
    a bernoulli arm, i.e., it has a reward of 1 with probability p and 0 with 
    probability (1-p). Now the agent has the dilemma of exploration vs exploitation.
    Now in e Greedy - 1, the agent will first explore in the first eT steps, that 
    exploration can be either deterministic or random, deterministic in the sense that
    we explore all arms equal number of times say in a round robin fashion, or in a random
    manner, i.e., we choose an arm randomly. After the exploration phase, the agent will
    exploit the best arm at each step, i.e., the arm with the highest empirical mean at 
    that step, not fixating on the best arm for the remaining (1-e)T steps like in 
    e Greedy - 1.
-------------------------------------------------------------------------------------- 

*/
int hor;
// This is a function that generates a bernoulli arm with a given probability p
int bernoulli_arm(double p) {
    random_device rd;  // Obtain a random number from hardware
    mt19937 eng(rd()); // Seed the generator
    bernoulli_distribution distr(p); // Define the distribution

    return distr(eng) ? 1 : 0; // Generate a random number based on the distribution
}

class arm {
public:
    double p; // Probability of success
    int count; // Number of times the arm has been pulled
    double reward; // Total reward received from this arm

    arm(double prob) : p(prob), count(0), reward(0.0) {}

    // Function to pull the arm and update the reward
    void pull() {
        int result = bernoulli_arm(p);
        count++;
        reward += result;
    }

    // Function to get the empirical mean of the arm
    double empirical_mean() const {
        return (count == 0) ? 0.0 : reward / count;
    }
};

double main3(double e = 0.06) { // This is the main function that implements the e Greedy - 2 algorithm with round robin exploration
    arm arms[] = {
        arm(0.1), // Arm 1 with probability 0.1
        arm(0.5), // Arm 2 with probability 0.5
        arm(0.9)  // Arm 3 with probability 0.9
    };
    int T = hor; // Total number of steps
    int exploration_steps = static_cast<int>(e * T);
    int exploitation_steps = T - exploration_steps;
    int num_arms = sizeof(arms) / sizeof(arms[0]);
    int arm_index = 0;
    // Exploration phase
    for (int i = 0; i < exploration_steps; i++) {
        arms[arm_index].pull(); // Pull the current arm
        arm_index = (arm_index + 1) % num_arms; // Move to the next arm
    }   
    // Exploitation phase, here we find the arm with the highest empirical mean and keep pulling it
    for (int i = 0; i < exploitation_steps; i++) {
        double best_mean = -1.0;
        int best_arm_index = -1;
        for (int j = 0; j < num_arms; j++) {
            double mean = arms[j].empirical_mean();
                if (mean > best_mean) {
                    best_mean = mean;
                    best_arm_index = j;
                }
        }
        arms[best_arm_index].pull(); // Pull the best arm

    }

    // Total reward from all arms
    double total_reward = 0.0;
    for (int j = 0; j < num_arms; j++) {
        total_reward += arms[j].reward;
    }

    cout << "Total reward from all arms: " << total_reward << endl;
    // cout << "------------------------------------------------------------" << endl;
    return total_reward; // Exit the program

}

double main2(double e = 0.06) { // This is the main function that implements the e Greedy - 1 algorithm with round robin exploration
    arm arms[] = {
        arm(0.1), // Arm 1 with probability 0.1
        arm(0.5), // Arm 2 with probability 0.5
        arm(0.9)  // Arm 3 with probability 0.9
    };
    int T = hor; // Total number of steps
    int exploration_steps = static_cast<int>(e * T);
    int exploitation_steps = T - exploration_steps;
    int num_arms = sizeof(arms) / sizeof(arms[0]);
    int arm_index = 0;
    // Exploration phase
    for (int i = 0; i < exploration_steps; i++) {
        arms[arm_index].pull(); // Pull the current arm
        arm_index = (arm_index + 1) % num_arms; // Move to the next arm
    }   
    // Exploitation phase, here we find the arm with the highest empirical mean and keep pulling it
    double best_mean = -1.0;
    int best_arm_index = -1;
    for (int j = 0; j < num_arms; j++) {
        double mean = arms[j].empirical_mean();
        if (mean > best_mean) {
            best_mean = mean;
            best_arm_index = j;
        }
    }
    
    for (int i = 0; i < exploitation_steps; i++) {

        arms[best_arm_index].pull(); // Pull the best arm

    }

    double total_reward = 0.0;
    for (int j = 0; j < num_arms; j++) {
        total_reward += arms[j].reward;
    }

    cout << "Best arm index: " << best_arm_index + 1 << " | Total reward from all arms: " << total_reward << endl;
    // cout << "------------------------------------------------------------" << endl;
    return total_reward; // Exit the program

}

int main() {
    cout << "\nEnter the horizon for the e Greedy algorithms: ";
    cin >> hor; // Input the horizon for the e Greedy algorithms
    cout << "------------------------------------------------------------" << endl;
    double e_1, e_2;
    cout << "Enter the exploration rate for e Greedy - 1 : "; 
    cin >> e_1 ; // Input the exploration rates for e Greedy - 1
    cout << "\nEnter the exploration rate for e Greedy - 2 : ";
    cin >> e_2 ; // Input the exploration rates for e Greedy - 2
    cout << '\n';
    cout << "------------------------------------------------------------------" << endl;
    cout << "Demonstrating e Greedy - 1 Algorithm with round robin exploration" << endl;
    cout << "------------------------------------------------------------------" << endl;
    double rew_eG1 = 0.0; // Initialize the total reward for e Greedy - 1
    for (int i = 0; i < 20; i++) {
        cout << "Run " << setw(2) << i + 1 << " : ";
        rew_eG1 += main2(e_1); // Call the main2 function multiple times to see different results
    }
    rew_eG1 /= 20; // Average the total reward over the runs

    cout << "------------------------------------------------------------------" << endl;
    cout << "Demonstrating e Greedy - 2 Algorithm with round robin exploration" << endl;
    cout << "------------------------------------------------------------------" << endl;
    double rew_eG2 = 0.0; // Initialize the total reward for e Greedy - 2
    for (int i = 0; i < 20; i++) {
        cout << "Run " << setw(2) << i + 1 << " : ";
        rew_eG2 += main3(e_2); // Call the main2 function multiple times to see different results
    }
    rew_eG2 /= 20; // Average the total reward over the runs

    cout << "------------------------------------------------------------------" << endl;
    cout << "Average Total Reward for e Greedy - 1: " << rew_eG1 << endl;
    cout << "Average Total Reward for e Greedy - 2: " << rew_eG2 << endl;
    cout << "------------------------------------------------------------------" << endl;
    return 0;
}