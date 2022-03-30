from unittest import result
import numpy as np
from enviroment import GraphicDisplay, Env

class PolicyIteration:
    def __init__(self,env):
        
        self.env=env
        self.value_table=[[0,0]*env.width for _ in range(env.height)]
        self.policy_table=[[[0.25,0.25,0.25,0.25]]*env.width for _ in range(env.height)]
        self.policy_table[2][2]=[]
        self.discount_factor=0.9
        
    def policy_evaluation(self):
        
        next_value_table=[[0.00]*self.env.width for _ in range(self.env.height)]
        for state in self.env.get_all_states():
            value=0.0
            if state ==[2,2]:
                next_value_table[state[0]][state[1]]=value
                continue
            
            for action in self.env.possible_action:
                
                next_state=self.env.state_after_action(state,action)
                reward=self.env.get_reward(state,action)
                next_value=self.get_value(next_state)
                value+=(self.get_policy(state)[action]*(reward+self.discount_factor*next_value))
                
            next_value_table[state[0]][state[1]]=value
        
        self.value_table=next_value_table
        
        
    def policy_improvement(self):
        
        next_policy=self.policy_table
        for state in self.env.get_all_states():
            if state ==[2,2]:
                continue
            
            value_list=[]
            result=[0.0,0.0,0.0,0.0]
            
            for index,action in enumerate(self.env.possible_action):
                next_state=self.env.state_after_action(state,action)
                reward=self.env.get_reward(state,action)
                next_value=self.get_value(next_state)
                value=reward+self.discount_factor*next_value
                value_list.append(value)
                
            max_idx_list=np.argwhere(value_list==np.amax(value_list))
            max_ids_list=max_idx_list.flatten().tolist()
            prob=1/len(max_idx_list)
            
            for idx in max_idx_list:
                result[idx]=prob
                
            next_policy[state[0]][state[1]]=result
            
        self.policy_table=next_policy
        
    def get_action(self,state):
        policy=self.get_policy(state)
        policy=np.array(policy)
        return np.random.choice(4,1,p=policy)[0]
    
    def get_policy(self,state):
        return self.policy_table[state[0]][state[1]]
    
    def get_value(self,state):
        return self.value_table[state[0]][state[1]]
    
    
env=Env()
policy_iteration=PolicyIteration(env)
grid_world=GraphicDisplay(policy_iteration)
grid_world.mainloop()