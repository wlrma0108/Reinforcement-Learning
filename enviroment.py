import tkinter as tk
from tkinter import Button, PhotoImage
import time
import numpy as np
from PIL import ImageTK,Image

PhotoImage = ImageTK,PhotoImage
UNIT=100
HEIGHT=5
WIDTH=5
TRANSITION_PROB=1
POSSIBLE_ACTION=[0,1,2,3]
ACTION=[(-1,0),(1,0),(0,-1),(0,1)]
REWARDS=[]

class GraphicDisplay(tk.Tk):
    def __init__(self,agent):
        super(GraphicDisplay,self).__init__()
        self.title('Policy Iteration')