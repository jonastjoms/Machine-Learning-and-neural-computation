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
% Remove comments to display
%disp("Value function: ")
%disp(" ")
% Calling print function for V
%format short
%print_V_table(V)

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
% Remove comments to display:
%disp("Likelihoods before and after policy optimisation: ")
%disp(" ")
%disp(likelihoods)

%% Question 4
% a)
% Generate trace with unbiased policy
% Remove comments here and in trace function to display:
% disp("Traces with unbiased policy:")
% disp(" ")

% Number of traces:
n_traces = 10;
% Generate traces, using a nested function.
[traces, all_rewards, all_actions] = generate_traces(n, a, T, R, Absorbing, Policy, n_traces);

% b)

% Generate the returns for every state from a given set of traces and
% and corresponding rewards.
returns = MC_policy_returns(n, traces, all_rewards, Absorbing, gamma);

% disp("Value function estimated with MC-First visit method:")
% disp(" ")
V_MC = MC_Value_function(returns);
% disp(V_MC)

% c)
% I use Mean Squared Error as measure of similarity between V and V_MC
MSE = [];
% Compute the distance for 1 to 10 traces and plot the result. Essentially 
% repeating the steps in b) but compute the distance every step.
% for n_traces = 1:10
%     [traces, all_rewards] = generate_traces(n, a, T, R, Absorbing, Policy, n_traces);
%     returns = MC_policy_returns(n, traces, all_rewards, Absorbing, gamma);
%     V_MC = MC_Value_function(returns);
%     MSE = [MSE, mean(sqrt((V-V_MC).^2))];
% end

% To see plot for 1 to 10 traces, remove comments below:
% n_traces = 1:10;
% figure
% plot(n_traces,MSE)
% grid on
% xlabel("Number of traces")
% ylabel("Mean Squared Error")
% title("MSE for MC First-Visit value function with respect to number of traces")

%% 5)
% Implementing epsilon-greedy policy for First-Fisit MC Control.

epsilon_1 = 0.1;
epsilon_2 = 0.75;

% Initializing parameters for e-soft algorithm:
% Toral state-action returns from all episodes:
total_s_a_returns = cell(n,a);

% Empty arrays for storing number of episodes, trace lengths and rewards:
n_episodes = [];
% List of trace lengths for every trial.
trace_lengths = [];
% List of reward sums for every trial
sum_rewards = [];

% Empty arrays used for averaging results
episodes_trace_length = [];
episodes_sum_rewards = [];

for episodes = 1:5:200
    % Initialize policy
    greedy_policy = Policy;
    
    % Do ten trials for every number of episodes:
    for trial = 1:10    
        % let the agent improve on a specific numbe rof episodes: 
        for k = 1:episodes
            [greedy_policy, states, rewards, actions] = MC_control(n, a, T, R, Absorbing, greedy_policy, gamma, epsilon_2);
        end
        % Store information from every trial:
        trace_lengths = [trace_lengths, length(states)];
        sum_rewards = [ sum_rewards, sum(rewards)];
    end
    % Average reward from ten trials and store information for current 
    % number of episodes:
    episodes_trace_length = [episodes_trace_length, mean(trace_lengths)];
    episodes_sum_rewards = [episodes_sum_rewards, mean(sum_rewards)];   
    n_episodes = [n_episodes, episodes];   
end
episodes_trace_length
episodes_sum_rewards

figure
plot(n_episodes,episodes_trace_length)
xlabel("Number of episodes")
ylabel("Trace length")
figure
plot(n_episodes,episodes_sum_rewards)
xlabel("Number of episodes")
ylabel("Reward")
   
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
function [traces, all_rewards, all_actions] = generate_traces(n, a, T, R, Absorbing, Policy, n_traces)
% Cell array for storing trace states and rewards, every cell represents 
% a trace and contains a list of states and rewards. 
traces = cell(1,n_traces); 
all_rewards = cell(1,n_traces);
all_actions = cell(1,n_traces);
% Create n_traces traces:
for i = 1:n_traces
    [states, rewards, actions] = generate_trace(n, a, T, R, Absorbing, Policy);
    traces{i} = states;
    all_rewards{i} = rewards;
    all_actions{i} = actions;
end
end

% Function for trace generation for given MDP
% Takes MDP information like number of states, actions, transistion matrix...
% reward matrix and absorbing states. In addition it  takes a policy.
% Returns, length of trace, the states in the trace as well as the rewards
function [states, rewards, actions] = generate_trace(n, a, T, R, Absorbing, Policy)
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
    % Printing trace:
    % Remove comments to display:
%     for state = 1:length(actions)
%         if rewards(state) == 0 | rewards(state) == -10
%             fprintf('s%d,%s,%d', states(state), actions(state),rewards(state))
%         else
%             fprintf('s%d,%s,%d,', states(state), actions(state),rewards(state))
%         end
%     end
%     disp(" ")
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
% improved policy, based on MDP episodes it generates.
function [greedy_policy, states, rewards, actions] = MC_control(n, a, T, R, Absorbing, Policy, gamma, epsilon)
    % Generate trace using current policy: 
    [states, rewards, actions] = generate_trace(n, a, T, R, Absorbing, Policy);
    % Obtain state-action returns for current trace
    state_action_returns = greedy_policy_returns(states, rewards, actions, a, n, gamma);
    % Append state-action returns to total returns:
    dim_returns = size(total_s_a_returns);
    % Iterate over all states:
    for i = 1:dim_returns(1)
        % And every action of every state:
        for j = 1:dim_returns(2)
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

        




