% Coursework in Machine Learning and Neural Computation
% Jonas Tjomsland, CID = 01570830
% To make it easier for the reader to understand I have tried to use
% similar set up as Dr. A. Aldo Faisal used for the first lab.

clc
clear all
close all
RunCoursework();

function RunCoursework()
%% Question 1

%Calculating persoonal p and personal gamma:
p = 0.5 + 0.5*(3/10);
gamma = 0.2 + 0.5*(0/10);


% Get system parameters from given grid world function
[NumStates, NumActions, TransitionMatrix, ...
 RewardMatrix, StateNames, ActionNames, AbsorbingStates] ...
 = PersonalisedGridWorld(p);

% Simplifying names:
n = NumStates;
a = NumActions;
T = TransitionMatrix;
R = RewardMatrix;
S = StateNames;
A = ActionNames;
Absorbing = AbsorbingStates;

% Creating policy matrix where the rows represent states and the columns
% possible actions: S1: N, E, S, W (14x4) matrix.
% Unbiased policy means equal probability of all actions.(1/4 in this case)
Policy = 1/4*ones(14,4);
% Chooosing tolerance for policy evaluation:
tol = 0.01;

%% Question 2

% Calling policy evaluation function:
V = policy_evaluation(n, a, T, R, Absorbing, Policy, tol, gamma);
disp("Value function: ")
disp(" ")
% Calling print function for V
format short
print_V_table(V)

%% Question 3 

% a) Likelihood

% Sequence vectors:
seq1 = [14, 10, 8, 4, 3];
seq2 = [11, 9, 5, 6, 6, 2];
seq3 = [12, 11, 11, 9, 5, 9,  5, 1, 2];

% Calling likelihood function:
likelihood1 = likelihood(seq1, T, Policy, a);
likelihood2 = likelihood(seq2, T, Policy, a);
likelihood3 = likelihood(seq3, T, Policy, a);

% b) Optimizing policy for likelihood

%Calling function for policy optimization:
optimal_policy = unbiased_policy(seq1, T, Policy, a);
optimal_policy = unbiased_policy(seq2, T, optimal_policy, a);
optimal_policy = unbiased_policy(seq3, T, optimal_policy, a);

% New likelihoods
likelihood1_improved = likelihood(seq1, T, optimal_policy, a);
likelihood2_improved = likelihood(seq2, T, optimal_policy, a);
likelihood3_improved = likelihood(seq3, T, optimal_policy, a);


% Print as table:
likelihoods = table(likelihood1, likelihood1_improved, likelihood2...
                    ,likelihood2_improved, likelihood3, likelihood3_improved);

disp("Likelihoods before and after policy optimisation: ")
disp(" ")
disp(likelihoods)

%% Question 4

% a)
% Generate trace with unbiased policy
% Remove comments here and in trace function to display:
disp("Traces with unbiased policy:")
disp(" ")
% Variable to let the functions know if we want to print:
print = 1;
% Number of traces:
n_traces = 10;
% Generate traces, using a nested function.
[traces, all_rewards, all_actions] = generate_traces(n, a, T, R, Absorbing, Policy, n_traces, print);
disp(" ")

% b)

% Generate the returns for every state from a given set of traces and
% and corresponding rewards.
returns = MC_policy_returns(n, traces, all_rewards, Absorbing, gamma);

disp("Value function estimated with MC-First visit method for 10 traces:")
disp(" ")
V_MC = MC_Value_function(returns);
format short
print_V_table(V_MC)

% c)
% I use Mean Squared Error as measure of similarity between V and V_MC
MSE = [];
% Variable to let the functions know if we want to print traces:
print = 0;
% Compute the distance for 1 to 10 traces and plot the result. Essentially 
% repeating the steps in b) but compute the distance every step.
for n_traces = 1:10
    [traces, all_rewards] = generate_traces(n, a, T, R, Absorbing, Policy, n_traces,print);
    returns = MC_policy_returns(n, traces, all_rewards, Absorbing, gamma);
    V_MC = MC_Value_function(returns);
    MSE = [MSE, mean(sqrt((V-V_MC).^2))];
end

%To see plot for 1 to 10 traces, remove comments below:
n_traces = 1:10;
figure
plot(n_traces,MSE)
grid on
xlabel("Number of traces")
ylabel("Mean Squared Error")
title("MSE for MC First-Visit value function with respect to number of traces")

%% Question 5

% Implementing epsilon-greedy policy for First-Fisit MC Control:
% Initialize epsilon 1 & 2
epsilon_1 = 0.1;
epsilon_2 = 0.75;
epsilon = [epsilon_1, epsilon_2];

% Max number of trials and episodes:
max_trials = 30;
max_episodes = 100;

% Create cell array to store trace length and rewards for every trial.
% Every cell represents 50 trials with a given number of episodes, e.g.
% cell ten in "trace_lengths" contains an array of 50 elements where 
% every element is the trace length from one trial with ten episodes.
% Double the numbers of cells and place the results for epsilon 2 after
% those from epsilon 1.
trace_lengths = cell(1, 2*max_episodes);
trials_rewards = cell(1, 2*max_episodes);

% For both epsilon:
for epsilon = epsilon
    % Do 20 trials for every number of episodes:
    for trials = 1:max_trials
        % Initiate policy as unbiased:
        greedy_policy = Policy;
        % Initiate empty cell matrix for state-action returns:
        total_s_a_returns = cell(n,a);
        % Let the agen operate for 1 to 200 episodes:
        for episodes = 1:max_episodes
            [greedy_policy, states, rewards, actions, total_s_a_returns] = MC_control(n, a, T, R, Absorbing, greedy_policy, gamma, epsilon, total_s_a_returns);
            % Append trace length and sum of rewards to cell array, place
            % results for epsilon 2 at cell 201-400:
            if epsilon == epsilon_2
                trace_lengths{max_episodes+episodes} = [trace_lengths{max_episodes+episodes}, length(states)];
                trials_rewards{max_episodes+episodes} = [trials_rewards{max_episodes+episodes}, sum(rewards)];
            else
                trace_lengths{episodes} = [trace_lengths{episodes}, length(states)];
                trials_rewards{episodes} = [trials_rewards{episodes}, sum(rewards)];
            end
        end
    end
end

% Array for averaged trace lengths and rewards as well as standard deviation:
mean_trace_lengths_e1 = [];
mean_rewards_e1 = [];
std_trace_lengths_e1 = [];
std_rewards_e1 = [];

mean_trace_lengths_e2 = [];
mean_rewards_e2 = [];
std_trace_lengths_e2 = [];
std_rewards_e2 = [];

% Compute mean and std for trace lengths and rewards for both espilon:
for i = 1:length(trace_lengths)
    % For epsilon 2:
    if i > length(trace_lengths)/2
        mean_trace_lengths_e2 = [mean_trace_lengths_e2, mean(trace_lengths{i})];
        mean_rewards_e2 = [ mean_rewards_e2, mean(trials_rewards{i})];
        std_trace_lengths_e2 = [std_trace_lengths_e2, std(trace_lengths{i})];
        std_rewards_e2 = [std_rewards_e2, std(trials_rewards{i})];
    % For epsilon 1:
    else
        mean_trace_lengths_e1 = [mean_trace_lengths_e1, mean(trace_lengths{i})];
        mean_rewards_e1 = [mean_rewards_e1, mean(trials_rewards{i})]; 
        std_trace_lengths_e1 = [std_trace_lengths_e1, std(trace_lengths{i})];
        std_rewards_e1 = [std_rewards_e1, std(trials_rewards{i})];        
    end
end

% Plotting results. First only the mean trace lengths adn rewards are
% plotted against episodes. Secondly mean plus/minus std are plotted. All
% for both epsilon 1 and epsilon 2.

% Mean trace length against episodes:
figure
semilogy(1:max_episodes,mean_trace_lengths_e1,'LineWidth',2)
hold on
semilogy(1:max_episodes,mean_trace_lengths_e2, 'LineWidth',2)
xlabel("Number of episodes")
ylabel("Trace lengths (log)")
legend("Epsilon = 0.1","Epsilon = 0.75")
title("Mean trace length in respect to episodes (30 trials)")

% Mean rewards against episodes:
figure
semilogy(1:max_episodes,mean_rewards_e1,'LineWidth',2)
hold on
semilogy(1:max_episodes,mean_rewards_e2, 'LineWidth',2)
xlabel("Number of episodes")
ylabel("Rewards (log)")
legend("Epsilon = 0.1","Epsilon = 0.75")
title("Mean rewards in respect to episodes (30 trials)")

% Mean trace lengths plus/minus standard deviation against episodes
figure
plot(1:max_episodes,mean_trace_lengths_e1-std_trace_lengths_e1, 'b','LineWidth',2)
hold on
plot(1:max_episodes,mean_trace_lengths_e2-std_trace_lengths_e2, 'r', 'LineWidth',2)
plot(1:max_episodes,mean_trace_lengths_e1+std_trace_lengths_e1,'b', 'LineWidth',2)
plot(1:max_episodes,mean_trace_lengths_e2+std_trace_lengths_e2, 'r', 'LineWidth',2)
xlabel("Number of episodes")
ylabel("Trace lengths")
legend("Epsilon = 0.1","Epsilon = 0.75")
title("Mean +- standard deviation of trace lengths with respect to episodes (30 trials)")

% Mean rewards plus/minus standard deviation against episodes
figure
plot(1:max_episodes,mean_rewards_e1-std_rewards_e1, 'b','LineWidth',2)
hold on
plot(1:max_episodes,mean_rewards_e2-std_rewards_e2, 'r', 'LineWidth',2)
plot(1:max_episodes,mean_rewards_e1+std_rewards_e1,'b', 'LineWidth',2)
plot(1:max_episodes,mean_rewards_e2+std_rewards_e2, 'r', 'LineWidth',2)
xlabel("Number of episodes")
ylabel("Rewards")
legend("Epsilon = 0.1","Epsilon = 0.75")
title("Mean +- standard deviation of rewards with respect to episodes (30 trials)")


%% Functions:

% Creating policy evaluation function
% Takes number of states, transition matrix, reward matrix, list of...
% absorbing states, policy and iteration tolerance as input.
% Returns value function V.
function V = policy_evaluation(n, a, T, R, Absorbing, Policy, tol, gamma)
    
    V = zeros(1,n);    % Optimal value function vector
    Vnew = V;       % Value function vector for step i+1.
    delta = 2*tol;  % Used to measure difference in V & Vnew.

    % Value iteration:
    while delta > tol
        for current_state = 1:n         % Iterating over all states
            if Absorbing(current_state) % Skipping terminal states
                continue
            end
            current_V = 0;              % Current value of current state
            for action = 1:a            % Iterating over all actions
                current_Q = 0;          % Current state action value 
                for next_state = 1:n    % Iterating over all states, the transition matrix will cancel oout states out of reach.
                    current_Q = current_Q + T(next_state, current_state, action)*(R(next_state, current_state, action) + gamma*V(next_state));
                end
                current_V = current_V + Policy(current_state, action)*current_Q;
            end
            Vnew(current_state) = current_V;  % Storing new value for current state
        end
    diff = abs(Vnew - V);  % Calculates changes in value function vector
    delta = max(diff);  % Compute new delta
    V = Vnew;              % Update value function
    end  
end

% Function for printing value function as table.  
% Takes value function vector as input and prints the table.
function print_V_table(V)
    
    % Creating table:
    State = "Value";
    S1 = V(1);
    S2 = V(2);
    S3 = V(3);
    S4 = V(4);
    S5 = V(5);
    S6 = V(6);
    S7 = V(7);
    S8 = V(8);
    S9 = V(9);
    S10 = V(10);
    S11 = V(11);
    S12 = V(12);
    S13 = V(13);
    S14 = V(14);
    Table = table(State,S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14);
    disp(Table)
end

% Function calculating likelihood of given sequence occuring.
% Takes sequence, transition matrix, policy and number of actions as input.
% Returns likelihood.
function likelihood = likelihood(seq, T, Policy, a)
    
    % Initializing total probability for iteration
    total_p = 1;
    % Iterating over the states of the sequence, stopping at last state
    for i = 1:length(seq)-1 
        current_state = seq(i);
        next_state = seq(i+1);
        % Initializing local probability for iteration  
        local_p = 0; 
        % Iterating over every action
        for action = 1:a 
            local_p = local_p + T(next_state, current_state, action)*Policy(current_state, action);
        end
        total_p = total_p*local_p;
    end
    likelihood = total_p*(1/4); % Multiply by 1/4 because that is the probability of starting in state 14
end
    
% Function optimizing the policy to increase the likelihood of observing a
% given sequence.
% Takes sequence, transition matrix, policy and number of actions as input.
% Returns optimal_policy.
function optimal_policy = unbiased_policy(seq, T, Policy, a)
    
    % Optimal policy matrix, just intend to change the policy of the states occuring in the sequence.
    optimal_policy = Policy;
    
    % Iterating over the states of the sequence, stopping at last state
    for i = 1:length(seq) 
        current_state = seq(i);
        % Setting no policy for terminal states
        if i == length(seq)
            optimal_policy(current_state, :) = 0;
            continue
        end
        next_state = seq(i+1);
        % Initializing local probability for iteration  
        local_p = []; 
        
        % Handle state 6 differently
        if current_state == 6 | current_state == 5
            optimal_action = 2;   % Picked East
            for action = 1:a
                if action == optimal_action
                    optimal_policy(current_state, action) = 1;
                else
                    optimal_policy(current_state, action) = 0;
                end 
            end
            continue    % Skipping to next state in the sequence
        end  

        % Iterating over every action
        for action = 1:a 
            local_p = [local_p, T(next_state, current_state, action)*Policy(current_state, action)];
        end

        [value, optimal_action] = max(local_p);
        for action = 1:a 
            if action == optimal_action
                optimal_policy(current_state, action) = 1;
            else
                optimal_policy(current_state, action) = 0;
            end
        end
    end
end

% Function that generates a given number of traces, it takes desired...
% number of traces and just calls trace generating function that amount of times.
% Returns a cell array of traces and rewards.
function [traces, all_rewards, all_actions] = generate_traces(n, a, T, R, Absorbing, Policy, n_traces, print)
% Cell array for storing trace states and rewards, every cell represents 
% a trace and contains a list of states and rewards. 
traces = cell(1,n_traces); 
all_rewards = cell(1,n_traces);
all_actions = cell(1,n_traces);
% Create n_traces traces:
for i = 1:n_traces
    if print
        fprintf('%d%s', i, ": ")
    end
    [states, rewards, actions] = generate_trace(n, a, T, R, Absorbing, Policy, print);
    traces{i} = states;
    all_rewards{i} = rewards;
    all_actions{i} = actions;
end
end

% Function for trace generation for given MDP
% Takes MDP information like number of states, actions, transistion matrix...
% reward matrix and absorbing states. In addition it  takes a policy.
% Returns, length of trace, the states in the trace as well as the rewards
function [states, rewards, actions] = generate_trace(n, a, T, R, Absorbing, Policy, print)
    % Start out by defining starting state and set that as current state:
    starting_states = [11, 12, 13 ,14];
    staring_states_p = (1/4)*ones(1,length(starting_states));
    current_state = randsrc(1, 1, [starting_states;staring_states_p]);
    % Array of the states visited, actions taken and rewards in the trace
    states = [current_state];
    action_strs = [];
    actions = [];
    rewards = [];
    % Initialize empyt trace array:
    trace = [];
    % Iterate until terminal state is reached:
    while 1
        % Decide action: (N=1, E=2, S=3, W=4)
        action = randsrc(1,1,[1:a; Policy(current_state,:)]);
        % Iterates over all states to decide next state.
        % Store prob for ending in state s in a array.
        next_state_p = [];
        for next_state = 1:n
            next_state_p = [ next_state_p,  T(next_state, current_state, action)];
        end
        % Choose next state:
        next_state = randsrc(1,1,[1:n; next_state_p]);
        % Define action as string:
        if action == 1
            action_str = "N";
        elseif action == 2
            action_str = "E";
        elseif action == 3
            action_str = "S";
        else
            action_str = "W";
        end
        reward = R(next_state,current_state,action);
        current_state = next_state;
        states = [states, current_state];  % Append to list of states
        action_strs = [action_strs, action_str];  % Append to list of action strings
        actions = [actions, action];
        rewards = [rewards, reward];
        if Absorbing(next_state)
            break
        end
    end
    if print
        %Printing trace:
        for state = 1:length(actions)
            if rewards(state) == 0 | rewards(state) == -10
                fprintf('S%d,%s,%d', states(state), action_strs(state),rewards(state))
            else
                fprintf('S%d,%s,%d,', states(state), action_strs(state),rewards(state))
            end
        end
        disp(" ")
    end
end

% Function that generate list of returns for every state for a given...
% number of traces. Takes a cell array of traces and a cell array of
% rewards. Calls "trace_returns" function for every trace.
function returns = MC_policy_returns(n, traces, all_rewards, Absorbing, gamma)
    
    % Empty returns cell array:
    returns = cell(1,n);
    % Iterate over all traces:
    for i = 1:length(traces)
        % Call function for Monte Carlo Policy Evaluation:
        trace_returns = MC_policy_evaluation(n, traces{i}, all_rewards{i}, Absorbing, gamma);
        % Append new returns to cell array:
        for j = 1:length(trace_returns)
            returns{j} = [returns{j} trace_returns{j}];
        end
    end
end

% Function which estimates value function based on observed episodes
% Takes number of states in MDP, an observed trace of states, the rewards
% obtained in that trace, information about terminal states and discount
% factor gamma.
function trace_returns = MC_policy_evaluation(n, trace, trace_rewards, Absorbing, gamma)  
    
        % Array for keeping track of visited states in trace:
        first_visit = ones(1,n);
        % Cell array for returns:
        trace_returns = cell(1,n);

        % Iterate over every state in trace:
        for i = 1:length(trace)
            % Skip terminal states:
            if Absorbing(trace(i))
                continue
            end
            % Only compute return if state hasn't been visited (First visit):
            if first_visit(trace(i))
                % Declare variable for discount:
                k = 0:(length(trace)-1-i); % (0 -> number of states left in trace)
                % Reward function:
                state_return = @(k) (gamma.^k).*trace_rewards(i:length(trace_rewards));
                state_return = sum(state_return(k));              
            else 
                continue
            end
            % Store that current state has been visited:
            first_visit(trace(i)) = 0;
            % Save state's return in list of state returns
            trace_returns{trace(i)} = state_return;
        end
end 
 
% Function that estimates value function given a list of discounted returns
% for every state.cReturns have previously been observed in a set of
% traces.
function V_MC = MC_Value_function(returns)
    % Arbitrary initial value function:
    V_MC = zeros(1,n);
    % Estimate value function:
    % Iterate over cells in returns
    for i = 1:length(returns)
        % Skip terminal states:
        if Absorbing(i)
            continue
        else
            if isempty(returns{i})
                 V_MC(i) = 0;
            else
                V_MC(i) = mean(returns{i});
            end
        end
    end
end

% Function for epsilon-soft algorithm for on-policy MC Control. 
% Takes information about MDP, a policy, epsilon and gamma. Returns an 
% improved policy, based on MDP episodes it generates and the updated list of returns.
function [greedy_policy, states, rewards, actions, total_s_a_returns] = MC_control(n, a, T, R, Absorbing, Policy, gamma, epsilon, total_s_a_returns)
    % Variable to let the functions know if we want to print:
    print = 0;
    % Generate trace using current policy: 
    [states, rewards, actions] = generate_trace(n, a, T, R, Absorbing, Policy, print);
    % Obtain state-action returns for current trace
    state_action_returns = greedy_policy_returns(states, rewards, actions, a, n, gamma);
    % Append state-action returns to total returns:
    % Iterate over all states:
    for i = 1:n
        % And every action of every state:
        for j = 1:a
            % Appending:
            total_s_a_returns{i,j} = [total_s_a_returns{i,j}, state_action_returns{i,j}];
        end
    end
    % Obtain new Q function:
    Q_MC = Q_MC_function(total_s_a_returns);
    % Update policy:
    for state = 1:length(Q_MC)
        [value, optimal_action] = max(Q_MC(state,:));
        for action = 1:a
            if action == optimal_action
                greedy_policy(state,action) = 1 - epsilon + (epsilon/a);
            else
                greedy_policy(state,action) = epsilon/a;
            end
        end
    end
end

% Function for state-action returns. Takes a list of states, rewards and
% actions from a trace. Also takes number of states and actions in MDP, 
% as well as information about terminal states and dicount factor. Returns 
% the discounted reward of every state-action pair.
function state_action_returns = greedy_policy_returns(states, rewards, actions, a, n, gamma)
        % Matrix for keeping track of visited state-action pairs in trace,
        % rows represent states and columns represent actions (1,2,3,4).
        first_visit = ones(n,a);
        % Cell matrix for returns:
        state_action_returns = cell(n,a);
        
        % Iterate over every state-action pair in the trace:
        for i = 1:length(actions)
            % Skip terminal states:
            if Absorbing(states(i))
                continue
            end    
         % Only compute return if state-action pair hasn't been visited:
            if first_visit(states(i),actions(i))  
                % Declare variable for discount:
                k = 0:(length(actions)-i); % (0 -> number of actions left in trace)
                % Reward function:
                state_action_return = @(k) (gamma.^k).*rewards(i:length(rewards));
                state_action_return = sum(state_action_return(k));              
            else 
                continue
            end
            % Store that current state-action pair has been visited:
            first_visit(states(i),actions(i)) = 0;
            % Store return in cell matrix
            state_action_returns{states(i),actions(i)} = state_action_return;
        end
end

% Function that takes a cell matrix of returns obtained from an unknown 
% number of traces and returns the Q function as the average return of
% every state-action pair.
function Q_MC = Q_MC_function(total_s_a_returns)
    % Arbitrary initial Q function:
    Q_MC = zeros(n ,a);
    % Estimate value function:
    % Iterate over cells in returns
    dim_returns = size(total_s_a_returns);
    % Iterate over all states
    for i = 1:(dim_returns(1)) 
        % Skip terminal states:
        if Absorbing(i)
            continue
        else
        % Iterate over all actions of current state
        for j = 1:(dim_returns(2))
            if isempty(total_s_a_returns{i,j})
                 Q_MC(i,j) = 0;
            else
                Q_MC(i,j) = mean(total_s_a_returns{i,j});
            end
        end
        end
    end
end

end

        




