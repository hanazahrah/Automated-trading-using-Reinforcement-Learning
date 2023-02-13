from importlib.resources import path
import gym
import sys
import torch
import pandas as pd
import Environment3_9_4h1h_revised_test2_limit
#from eval_policy import eval
#from arguments import get_args
#from network2 import FeedForwardNN
import numpy as np
from torch.distributions import Categorical
import tensorflow as tf
import numpy as np
import pickle

from Environment3_9_4h1h_revised_test2_limit import BUY, HOLD, SELL

class QAgent:
    def __init__(self, state_size, action_size):
        self.EPSILON = .1
        self.ALPHA = .85
        self.GAMMA = .99
        self.num_action = action_size
        self.Q = np.zeros([10, 3])

    def chooseAction(self, state):
        if np.random.binomial(1, self.EPSILON) == 1:
            return np.random.choice(self.num_action)
        else:
            return np.argmax(self.Q[state,:])
    
    def learn(self, state, action, reward, nextState):
        self.Q[state, action] += self.ALPHA*(reward + self.GAMMA*np.max(self.Q[nextState,:]) - self.Q[state, action])

    def reduceExploration(self, i):
            self.EPSILON /= i+1

def q_state7(position, reward, n_hold, n_delta, color, colorH, colorH2):
  state = 0
  if position == 0:
    if color == 0:
      if reward == 0:
        if n_delta < 288:#2015:
          if position == colorH: 
            if position == colorH2: state = 1
            else: state = 2
          else: 
            if position == colorH2: state = 3
            else: state = 4
        else:
          if position == colorH: 
            if position == colorH2: state = 5
            else: state = 6
          else: 
            if position == colorH2: state = 7
            else: state = 8
      else:
        state = 9
    elif color == 1:
      if reward == 0:
        if n_delta < 288:
          if position == colorH: 
            if position == colorH2: state = 10
            else: state = 11
          else: 
            if position == colorH2: state = 12
            else: state = 13
        else:
          if position == colorH: 
            if position == colorH2: state = 14
            else: state = 15
          else: 
            if position == colorH2: state = 16
            else: state = 17
      else:
        state = 18
    elif color == -1:
      if reward == 0:
        if n_delta < 288:#2015:
          if position == colorH: 
            if position == colorH2: state = 19
            else: state = 20
          else: 
            if position == colorH2: state = 21
            else: state = 22
        else:
          if position == colorH: 
            if position == colorH2: state = 23
            else: state = 24
          else: 
            if position == colorH2: state = 25
            else: state = 26
      else:
        state = 27
  elif position == 1:
    if color == 0:
      if reward >= 0:
        if n_hold < 288:
          if position == colorH: 
            if position == colorH2: state = 28
            else: state = 29
          else: 
            if position == colorH2: state = 30
            else: state = 31
        else:
          if position == colorH: 
            if position == colorH2: state = 32
            else: state = 33
          else: 
            if position == colorH2: state = 34
            else: state = 35
      elif reward < 0:
        if n_hold < 288:
          if position == colorH: 
            if position == colorH2: state = 36
            else: state = 37
          else: 
            if position == colorH2: state = 38
            else: state = 39
        else:
          if position == colorH: 
            if position == colorH2: state = 40
            else: state = 41
          else: 
            if position == colorH2: state = 42
            else: state = 43
    elif color == 1:
      if reward >= 0:
        if n_hold < 288:
          if position == colorH: 
            if position == colorH2: state = 44
            else: state = 45
          else: 
            if position == colorH2: state = 46
            else: state = 47
        else:
          if position == colorH: 
            if position == colorH2: state = 48
            else: state = 49
          else: 
            if position == colorH2: state = 50
            else: state = 51
      elif reward < 0:
        if n_hold < 288:
          if position == colorH: 
            if position == colorH2: state = 52
            else: state = 53
          else: 
            if position == colorH2: state = 54
            else: state = 55
        else:
          if position == colorH: 
            if position == colorH2: state = 56
            else: state = 57
          else: 
            if position == colorH2: state = 58
            else: state = 59
    elif color == -1:
      if reward >= 0:
        if n_hold < 288:
          if position == colorH: 
            if position == colorH2: state = 60
            else: state = 61
          else: 
            if position == colorH2: state = 62
            else: state = 63
        else:
          if position == colorH: 
            if position == colorH2: state = 64
            else: state = 65
          else: 
            if position == colorH2: state = 66
            else: state = 67
      elif reward < 0:
        if n_hold < 288:
          if position == colorH: 
            if position == colorH2: state = 68
            else: state = 69
          else: 
            if position == colorH2: state = 70
            else: state = 71
        else:
          if position == colorH: 
            if position == colorH2: state = 72
            else: state = 73
          else: 
            if position == colorH2: state = 74
            else: state = 75
  elif position == -1:
    if color == 0:
      if reward >= 0:
        if n_hold < 288:
          if position == colorH: 
            if position == colorH2: state = 76
            else: state = 77
          else: 
            if position == colorH2: state = 78
            else: state = 79
        else:
          if position == colorH: 
            if position == colorH2: state = 80
            else: state = 81
          else: 
            if position == colorH2: state = 82
            else: state = 83
      elif reward < 0:
        if n_hold < 288:
          if position == colorH: 
            if position == colorH2: state = 84
            else: state = 85
          else: 
            if position == colorH2: state = 86
            else: state = 87
        else:
          if position == colorH: 
            if position == colorH2: state = 88
            else: state = 89
          else: 
            if position == colorH2: state = 90
            else: state = 91
    elif color == 1:
      if reward >= 0:
        if n_hold < 288:
          if position == colorH: 
            if position == colorH2: state = 92
            else: state = 93
          else: 
            if position == colorH2: state = 94
            else: state = 95
        else:
          if position == colorH: 
            if position == colorH2: state = 96
            else: state = 97
          else: 
            if position == colorH2: state = 98
            else: state = 99
      elif reward < 0:
        if n_hold < 288:
          if position == colorH: 
            if position == colorH2: state = 100
            else: state = 101
          else: 
            if position == colorH2: state = 102
            else: state = 103
        else:
          if position == colorH: 
            if position == colorH2: state = 104
            else: state = 105
          else: 
            if position == colorH2: state = 106
            else: state = 107
    elif color == -1:
      if reward >= 0:
        if n_hold < 288:
          if position == colorH: 
            if position == colorH2: state = 108
            else: state = 109
          else: 
            if position == colorH2: state = 110
            else: state = 111
        else:
          if position == colorH: 
            if position == colorH2: state = 112
            else: state = 113
          else: 
            if position == colorH2: state = 114
            else: state = 115
      elif reward < 0:
        if n_hold < 288:
          if position == colorH: 
            if position == colorH2: state = 116
            else: state = 117
          else: 
            if position == colorH2: state = 118
            else: state = 119
        else:
          if position == colorH: 
            if position == colorH2: state = 120
            else: state = 121
          else: 
            if position == colorH2: state = 122
            else: state = 123
  return state

def q_state14(position, reward, profit, n_hold, n_delta, color, colorH, colorH2):
    state = 0
    if position == 0:
        if color == 0:
            if reward == 0:
                if n_delta <= 5:
                    if position == colorH: 
                        if position == colorH2: state = 1
                        else: state = 2
                    else: 
                        if position == colorH2: state = 3
                        else: state = 4
                elif n_delta > 5 and n_delta <=12:
                    if position == colorH: 
                        if position == colorH2: state = 5
                        else: state = 6
                    else: 
                        if position == colorH2: state = 7
                        else: state = 8
                elif n_delta > 12 and n_delta <= 48:
                    if position == colorH: 
                        if position == colorH2: state = 9
                        else: state = 10
                    else: 
                        if position == colorH2: state = 11
                        else: state = 12
                else: 
                    if position == colorH: 
                        if position == colorH2: state = 13
                        else: state = 14
                    else: 
                        if position == colorH2: state = 15
                        else: state = 16
            else:
                state = 17
        elif color == 1:
            if reward == 0:
                if n_delta <= 5:
                    if position == colorH: 
                        if position == colorH2: state = 18
                        else: state = 19
                    else: 
                        if position == colorH2: state = 20
                        else: state = 21
                elif n_delta > 5 and n_delta<=12:
                    if position == colorH: 
                        if position == colorH2: state = 22
                        else: state = 23
                    else: 
                        if position == colorH2: state = 24
                        else: state = 25
                elif n_delta > 12 and n_delta<= 48:
                    if position == colorH: 
                        if position == colorH2: state = 26
                        else: state = 27
                    else: 
                        if position == colorH2: state = 28
                        else: state = 29
                else: 
                    if position == colorH: 
                        if position == colorH2: state = 30
                        else: state = 31
                    else: 
                        if position == colorH2: state = 32
                        else: state = 33
            else:
                state = 34
        elif color == -1:
            if reward == 0:
                if n_delta <= 5:
                    if position == colorH: 
                        if position == colorH2: state = 35
                        else: state = 36
                    else: 
                        if position == colorH2: state = 37
                        else: state = 38
                elif n_delta > 5 and n_delta<=12:
                    if position == colorH: 
                        if position == colorH2: state = 39
                        else: state = 40
                    else: 
                        if position == colorH2: state = 41
                        else: state = 42
                elif n_delta > 12 and n_delta<= 48:
                    if position == colorH: 
                        if position == colorH2: state = 43
                        else: state = 44
                    else: 
                        if position == colorH2: state = 45
                        else: state = 46
                else: 
                    if position == colorH: 
                        if position == colorH2: state = 47
                        else: state = 48
                    else: 
                        if position == colorH2: state = 49
                        else: state = 50
            else:
                state = 51
    elif position == 1:
        if color == 0:
            if reward >= 0:
                if n_hold < 288:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 52
                            else: state = 53
                        else: 
                            if position == colorH2: state = 54
                            else: state = 55
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 56
                            else: state = 57
                        else: 
                            if position == colorH2: state = 58
                            else: state = 59
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 60
                            else: state = 61
                        else: 
                            if position == colorH2: state = 62
                            else: state = 63 
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 64
                            else: state = 65
                        else: 
                            if position == colorH2: state = 66
                            else: state = 67
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 68
                            else: state = 69
                        else: 
                            if position == colorH2: state = 70
                            else: state = 71
                else:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 72
                            else: state = 73
                        else: 
                            if position == colorH2: state = 74
                            else: state = 75
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 76
                            else: state = 77
                        else: 
                            if position == colorH2: state = 78
                            else: state = 79
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 80
                            else: state = 81
                        else: 
                            if position == colorH2: state = 82
                            else: state = 83
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 84
                            else: state = 85
                        else: 
                            if position == colorH2: state = 86
                            else: state = 87
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 88
                            else: state = 89
                        else: 
                            if position == colorH2: state = 90
                            else: state = 91
            elif reward < 0:
                if n_hold < 288:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 92
                            else: state = 93
                        else: 
                            if position == colorH2: state = 94
                            else: state = 95
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 96
                            else: state = 97
                        else: 
                            if position == colorH2: state = 98
                            else: state = 99
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 100
                            else: state = 101
                        else: 
                            if position == colorH2: state = 102
                            else: state = 103
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 104
                            else: state = 105
                        else: 
                            if position == colorH2: state = 106
                            else: state = 107
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 108
                            else: state = 109
                        else: 
                            if position == colorH2: state = 110
                            else: state = 111
                else:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 112
                            else: state = 113
                        else: 
                            if position == colorH2: state = 114
                            else: state = 115
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 116
                            else: state = 117
                        else: 
                            if position == colorH2: state = 118
                            else: state = 119
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 120
                            else: state = 121
                        else: 
                            if position == colorH2: state = 122
                            else: state = 123
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 124
                            else: state = 125
                        else: 
                            if position == colorH2: state = 126
                            else: state = 127
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 128
                            else: state = 129
                        else: 
                            if position == colorH2: state = 130
                            else: state = 131
        elif color == 1:
            if reward >= 0:
                if n_hold < 288:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 132
                            else: state = 133
                        else: 
                            if position == colorH2: state = 134
                            else: state = 135
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 136
                            else: state = 137
                        else: 
                            if position == colorH2: state = 138
                            else: state = 139
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 140
                            else: state = 141
                        else: 
                            if position == colorH2: state = 142
                            else: state = 143
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 144
                            else: state = 145
                        else: 
                            if position == colorH2: state = 146
                            else: state = 147
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 148
                            else: state = 149
                        else: 
                            if position == colorH2: state = 150
                            else: state = 151
                else:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 152
                            else: state = 153
                        else: 
                            if position == colorH2: state = 154
                            else: state = 155
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 156
                            else: state = 157
                        else: 
                            if position == colorH2: state = 158
                            else: state = 159
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 160
                            else: state = 161
                        else: 
                            if position == colorH2: state = 162
                            else: state = 163
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 164
                            else: state = 165
                        else: 
                            if position == colorH2: state = 166
                            else: state = 167
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 168
                            else: state = 169
                        else: 
                            if position == colorH2: state = 170
                            else: state = 171
            elif reward < 0:
                if n_hold < 288:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 172
                            else: state = 173
                        else: 
                            if position == colorH2: state = 174
                            else: state = 175
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 176
                            else: state = 177
                        else: 
                            if position == colorH2: state = 178
                            else: state = 179
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 180
                            else: state = 181
                        else: 
                            if position == colorH2: state = 182
                            else: state = 183
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 184
                            else: state = 185
                        else: 
                            if position == colorH2: state = 186
                            else: state = 187
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 188
                            else: state = 189
                        else: 
                            if position == colorH2: state = 190
                            else: state = 191
                else:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 192
                            else: state = 193
                        else: 
                            if position == colorH2: state = 194
                            else: state = 195
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 196
                            else: state = 197
                        else: 
                            if position == colorH2: state = 198
                            else: state = 199
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 200
                            else: state = 201
                        else: 
                            if position == colorH2: state = 202
                            else: state = 203
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 204
                            else: state = 205
                        else: 
                            if position == colorH2: state = 206
                            else: state = 207
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 208
                            else: state = 209
                        else: 
                            if position == colorH2: state = 210
                            else: state = 211
        elif color == -1:
            if reward >= 0:
                if n_hold < 288:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 212
                            else: state = 213
                        else: 
                            if position == colorH2: state = 214
                            else: state = 215
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 216
                            else: state = 217
                        else: 
                            if position == colorH2: state = 218
                            else: state = 219
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 220
                            else: state = 221
                        else: 
                            if position == colorH2: state = 222
                            else: state = 223
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 224
                            else: state = 225
                        else: 
                            if position == colorH2: state = 226
                            else: state = 227
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 228
                            else: state = 229
                        else: 
                            if position == colorH2: state = 230
                            else: state = 231
                else:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 232
                            else: state = 233
                        else: 
                            if position == colorH2: state = 234
                            else: state = 235
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 236
                            else: state = 237
                        else: 
                            if position == colorH2: state = 238
                            else: state = 239
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 240
                            else: state = 241
                        else: 
                            if position == colorH2: state = 242
                            else: state = 243
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 244
                            else: state = 245
                        else: 
                            if position == colorH2: state = 246
                            else: state = 247
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 248
                            else: state = 249
                        else: 
                            if position == colorH2: state = 250
                            else: state = 251
            elif reward < 0:
                if n_hold < 288:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 252
                            else: state = 253
                        else: 
                            if position == colorH2: state = 254
                            else: state = 255
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 256
                            else: state = 257
                        else: 
                            if position == colorH2: state = 258
                            else: state = 259
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 260
                            else: state = 261
                        else: 
                            if position == colorH2: state = 262
                            else: state = 263
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 264
                            else: state = 265
                        else: 
                            if position == colorH2: state = 266
                            else: state = 267
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 268
                            else: state = 269
                        else: 
                            if position == colorH2: state = 270
                            else: state = 271
                else:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 272
                            else: state = 273
                        else: 
                            if position == colorH2: state = 274
                            else: state = 275
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 276
                            else: state = 277
                        else: 
                            if position == colorH2: state = 278
                            else: state = 279
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 280
                            else: state = 281
                        else: 
                            if position == colorH2: state = 282
                            else: state = 283
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 284
                            else: state = 285
                        else: 
                            if position == colorH2: state = 286
                            else: state = 287
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 288
                            else: state = 289
                        else: 
                            if position == colorH2: state = 290
                            else: state = 291
    elif position == -1:
        if color == 0:
            if reward >= 0:
                if n_hold < 288:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 292
                            else: state = 293
                        else: 
                            if position == colorH2: state = 294
                            else: state = 295
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 296
                            else: state = 297
                        else: 
                            if position == colorH2: state = 298
                            else: state = 299
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 300
                            else: state = 301
                        else: 
                            if position == colorH2: state = 302
                            else: state = 303
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 304
                            else: state = 305
                        else: 
                            if position == colorH2: state = 306
                            else: state = 307
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 308
                            else: state = 309
                        else: 
                            if position == colorH2: state = 310
                            else: state = 311
                else:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 312
                            else: state = 313
                        else: 
                            if position == colorH2: state = 314
                            else: state = 315
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 316
                            else: state = 317
                        else: 
                            if position == colorH2: state = 318
                            else: state = 319
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 320
                            else: state = 321
                        else: 
                            if position == colorH2: state = 322
                            else: state = 323
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 324
                            else: state = 325
                        else: 
                            if position == colorH2: state = 326
                            else: state = 327
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 328
                            else: state = 329
                        else: 
                            if position == colorH2: state = 330
                            else: state = 331
            elif reward < 0:
                if n_hold < 288:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 332
                            else: state = 333
                        else: 
                            if position == colorH2: state = 334
                            else: state = 335
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 336
                            else: state = 337
                        else: 
                            if position == colorH2: state = 338
                            else: state = 339
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 340
                            else: state = 341
                        else: 
                            if position == colorH2: state = 342
                            else: state = 343
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 344
                            else: state = 345
                        else: 
                            if position == colorH2: state = 346
                            else: state = 347
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 348
                            else: state = 349
                        else: 
                            if position == colorH2: state = 350
                            else: state = 351
                else:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 352
                            else: state = 353
                        else: 
                            if position == colorH2: state = 354
                            else: state = 355
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 356
                            else: state = 357
                        else: 
                            if position == colorH2: state = 358
                            else: state = 359
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 360
                            else: state = 361
                        else: 
                            if position == colorH2: state = 362
                            else: state = 363
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 364
                            else: state = 365
                        else: 
                            if position == colorH2: state = 366
                            else: state = 367
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 368
                            else: state = 369
                        else: 
                            if position == colorH2: state = 370
                            else: state = 371
        elif color == 1:
            if reward >= 0:
                if n_hold < 288:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 372
                            else: state = 373
                        else: 
                            if position == colorH2: state = 374
                            else: state = 375
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 376
                            else: state = 377
                        else: 
                            if position == colorH2: state = 378
                            else: state = 379
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 380
                            else: state = 381
                        else: 
                            if position == colorH2: state = 382
                            else: state = 383
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 384
                            else: state = 385
                        else: 
                            if position == colorH2: state = 386
                            else: state = 387
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 388
                            else: state = 389
                        else: 
                            if position == colorH2: state = 390
                            else: state = 391
                else:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 392
                            else: state = 393
                        else: 
                            if position == colorH2: state = 394
                            else: state = 395
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 396
                            else: state = 397
                        else: 
                            if position == colorH2: state = 398
                            else: state = 399
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 400
                            else: state = 401
                        else: 
                            if position == colorH2: state = 402
                            else: state = 403
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 404
                            else: state = 405
                        else: 
                            if position == colorH2: state = 406
                            else: state = 407
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 408
                            else: state = 409
                        else: 
                            if position == colorH2: state = 410
                            else: state = 411
            elif reward < 0:
                if n_hold < 288:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 412
                            else: state = 413
                        else: 
                            if position == colorH2: state = 414
                            else: state = 415
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 416
                            else: state = 417
                        else: 
                            if position == colorH2: state = 418
                            else: state = 419
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 420
                            else: state = 421
                        else: 
                            if position == colorH2: state = 422
                            else: state = 423
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 424
                            else: state = 425
                        else: 
                            if position == colorH2: state = 426
                            else: state = 427
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 428
                            else: state = 429
                        else: 
                            if position == colorH2: state = 430
                            else: state = 431
                else:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 432
                            else: state = 433
                        else: 
                            if position == colorH2: state = 434
                            else: state = 435
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 436
                            else: state = 437
                        else: 
                            if position == colorH2: state = 438
                            else: state = 439
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 440
                            else: state = 441
                        else: 
                            if position == colorH2: state = 442
                            else: state = 443
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 444
                            else: state = 445
                        else: 
                            if position == colorH2: state = 446
                            else: state = 447
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 448
                            else: state = 449
                        else: 
                            if position == colorH2: state = 450
                            else: state = 451
        elif color == -1:
            if reward >= 0:
                if n_hold < 288:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 452
                            else: state = 453
                        else: 
                            if position == colorH2: state = 454
                            else: state = 455
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 456
                            else: state = 457
                        else: 
                            if position == colorH2: state = 458
                            else: state = 459
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 460
                            else: state = 461
                        else: 
                            if position == colorH2: state = 462
                            else: state = 463
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 464
                            else: state = 465
                        else: 
                            if position == colorH2: state = 466
                            else: state = 467
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 468
                            else: state = 469
                        else: 
                            if position == colorH2: state = 470
                            else: state = 471
                else:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 472
                            else: state = 473
                        else: 
                            if position == colorH2: state = 474
                            else: state = 475
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 476
                            else: state = 477
                        else: 
                            if position == colorH2: state = 478
                            else: state = 479
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 480
                            else: state = 481
                        else: 
                            if position == colorH2: state = 482
                            else: state = 483
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 484
                            else: state = 485
                        else: 
                            if position == colorH2: state = 486
                            else: state = 487
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 488
                            else: state = 489
                        else: 
                            if position == colorH2: state = 490
                            else: state = 491
            elif reward < 0:
                if n_hold < 288:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 492
                            else: state = 493
                        else: 
                            if position == colorH2: state = 494
                            else: state = 495
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 496
                            else: state = 497
                        else: 
                            if position == colorH2: state = 498
                            else: state = 499
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 500
                            else: state = 501
                        else: 
                            if position == colorH2: state = 502
                            else: state = 503
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 504
                            else: state = 505
                        else: 
                            if position == colorH2: state = 506
                            else: state = 507
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 508
                            else: state = 509
                        else: 
                            if position == colorH2: state = 510
                            else: state = 511
                else:
                    if profit <= 3:
                        if position == colorH: 
                            if position == colorH2: state = 512
                            else: state = 513
                        else: 
                            if position == colorH2: state = 514
                            else: state = 515
                    elif profit > 3 and profit <= 10:
                        if position == colorH: 
                            if position == colorH2: state = 516
                            else: state = 517
                        else: 
                            if position == colorH2: state = 518
                            else: state = 519
                    elif profit > 10 and profit <= 17:
                        if position == colorH: 
                            if position == colorH2: state = 520
                            else: state = 521
                        else: 
                            if position == colorH2: state = 522
                            else: state = 523
                    elif profit > 17 and profit <= 27:
                        if position == colorH: 
                            if position == colorH2: state = 524
                            else: state = 525
                        else: 
                            if position == colorH2: state = 526
                            else: state = 527
                    else:
                        if position == colorH: 
                            if position == colorH2: state = 528
                            else: state = 529
                        else: 
                            if position == colorH2: state = 530
                            else: state = 531
    return state

def q_state12(position, reward, profit, n_hold, n_delta, color, colorH, colorH2):
  state = 0
  if position == 0:
    if color == 0:
      if reward == 0:
        if n_delta < 288:
          if position == colorH: 
            if position == colorH2: state = 1
            else: state = 2
          else: 
            if position == colorH2: state = 3
            else: state = 4
        else:
          if position == colorH: 
            if position == colorH2: state = 5
            else: state = 6
          else: 
            if position == colorH2: state = 7
            else: state = 8
      else:
        state = 9
    elif color == 1:
      if reward == 0:
        if n_delta < 288:
          if position == colorH: 
            if position == colorH2: state = 10
            else: state = 11
          else: 
            if position == colorH2: state = 12
            else: state = 13
        else:
          if position == colorH: 
            if position == colorH2: state = 14
            else: state = 15
          else: 
            if position == colorH2: state = 16
            else: state = 17
      else:
        state = 18
    elif color == -1:
      if reward == 0:
        if n_delta < 288:
          if position == colorH: 
            if position == colorH2: state = 19 
            else: state = 20
          else: 
            if position == colorH2: state = 21
            else: state = 22
        else:
          if position == colorH: 
            if position == colorH2: state = 23
            else: state = 24
          else: 
            if position == colorH2: state = 25
            else: state = 26
      else:
        state = 27
  elif position == 1:
    if color == 0:
      if reward >= 0:
        if n_hold < 288:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 28
              else: state = 29
            else: 
              if position == colorH2: state = 30
              else: state = 31
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 32
              else: state = 33
            else: 
              if position == colorH2: state = 34
              else: state = 35
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 36
              else: state = 37
            else: 
              if position == colorH2: state = 38
              else: state = 39
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 40
              else: state = 41
            else: 
              if position == colorH2: state = 42
              else: state = 43
          else:
            if position == colorH: 
              if position == colorH2: state = 44
              else: state = 45
            else: 
              if position == colorH2: state = 46
              else: state = 47
        else:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 48
              else: state = 49
            else: 
              if position == colorH2: state = 50
              else: state = 51
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 52
              else: state = 53
            else: 
              if position == colorH2: state = 54
              else: state = 55
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 56
              else: state = 57
            else: 
              if position == colorH2: state = 58
              else: state = 59
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 60
              else: state = 61
            else: 
              if position == colorH2: state = 62
              else: state = 63
          else:
            if position == colorH: 
              if position == colorH2: state = 64
              else: state = 65
            else: 
              if position == colorH2: state = 66
              else: state = 67
      elif reward < 0:
        if n_hold < 288:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 68
              else: state = 69
            else: 
              if position == colorH2: state = 70
              else: state = 71
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 72
              else: state = 73
            else: 
              if position == colorH2: state = 74
              else: state = 75
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 76
              else: state = 77
            else: 
              if position == colorH2: state = 78
              else: state = 79
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 80
              else: state = 81
            else: 
              if position == colorH2: state = 82
              else: state = 83
          else:
            if position == colorH: 
              if position == colorH2: state = 84
              else: state = 85
            else: 
              if position == colorH2: state = 86
              else: state = 87
        else:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 88
              else: state = 89
            else: 
              if position == colorH2: state = 90
              else: state = 91
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 92
              else: state = 93
            else: 
              if position == colorH2: state = 94
              else: state = 95
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 96
              else: state = 97
            else: 
              if position == colorH2: state = 98
              else: state = 99
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 100
              else: state = 101
            else: 
              if position == colorH2: state = 102
              else: state = 103
          else:
            if position == colorH: 
              if position == colorH2: state = 104
              else: state = 105
            else: 
              if position == colorH2: state = 106
              else: state = 107
    elif color == 1:
      if reward >= 0:
        if n_hold < 288:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 108
              else: state = 109
            else: 
              if position == colorH2: state = 110
              else: state = 111
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 112
              else: state = 113
            else: 
              if position == colorH2: state = 114
              else: state = 115
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 116
              else: state = 117
            else: 
              if position == colorH2: state = 118
              else: state = 119
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 120
              else: state = 121
            else: 
              if position == colorH2: state = 122
              else: state = 123
          else:
            if position == colorH: 
              if position == colorH2: state = 124
              else: state = 125
            else: 
              if position == colorH2: state = 126
              else: state = 127
        else:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 128
              else: state = 129
            else: 
              if position == colorH2: state = 130
              else: state = 131
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 132
              else: state = 133
            else: 
              if position == colorH2: state = 134
              else: state = 135
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 136
              else: state = 137
            else: 
              if position == colorH2: state = 138
              else: state = 139
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 140
              else: state = 141
            else: 
              if position == colorH2: state = 142
              else: state = 143
          else:
            if position == colorH: 
              if position == colorH2: state = 144
              else: state = 145
            else: 
              if position == colorH2: state = 146
              else: state = 147
      elif reward < 0:
        if n_hold < 288:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 148
              else: state = 149
            else: 
              if position == colorH2: state = 150
              else: state = 151
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 152
              else: state = 153
            else: 
              if position == colorH2: state = 154
              else: state = 155
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 156
              else: state = 157
            else: 
              if position == colorH2: state = 158
              else: state = 159
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 160
              else: state = 161
            else: 
              if position == colorH2: state = 162
              else: state = 163
          else:
            if position == colorH: 
              if position == colorH2: state = 164
              else: state = 165
            else: 
              if position == colorH2: state = 166
              else: state = 167
        else:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 168
              else: state = 169
            else: 
              if position == colorH2: state = 170
              else: state = 171
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 172
              else: state = 173
            else: 
              if position == colorH2: state = 174
              else: state = 175
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 176
              else: state = 177
            else: 
              if position == colorH2: state = 178
              else: state = 179
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 180
              else: state = 181
            else: 
              if position == colorH2: state = 182
              else: state = 183
          else:
            if position == colorH: 
              if position == colorH2: state = 184
              else: state = 185
            else: 
              if position == colorH2: state = 186
              else: state = 187
    elif color == -1:
      if reward >= 0:
        if n_hold < 288:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 188
              else: state = 189
            else: 
              if position == colorH2: state = 190
              else: state = 191
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 192
              else: state = 193
            else: 
              if position == colorH2: state = 194
              else: state = 195
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 196
              else: state = 197
            else: 
              if position == colorH2: state = 198
              else: state = 199
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 200
              else: state = 201
            else: 
              if position == colorH2: state = 202
              else: state = 43
          else:
            if position == colorH: 
              if position == colorH2: state = 44
              else: state = 205
            else: 
              if position == colorH2: state = 206
              else: state = 207
        else:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 208
              else: state = 209
            else: 
              if position == colorH2: state = 210
              else: state = 211
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 212
              else: state = 213
            else: 
              if position == colorH2: state = 214
              else: state = 215
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 216
              else: state = 217
            else: 
              if position == colorH2: state = 218
              else: state = 219
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 220
              else: state = 221
            else: 
              if position == colorH2: state = 222
              else: state = 223
          else:
            if position == colorH: 
              if position == colorH2: state = 224
              else: state = 225
            else: 
              if position == colorH2: state = 226
              else: state = 227
      elif reward < 0:
        if n_hold < 288:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 228
              else: state = 229
            else: 
              if position == colorH2: state = 230
              else: state = 231
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 232
              else: state = 233
            else: 
              if position == colorH2: state = 234
              else: state = 235
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 236
              else: state = 237
            else: 
              if position == colorH2: state = 238
              else: state = 239
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 240
              else: state = 241
            else: 
              if position == colorH2: state = 242
              else: state = 243
          else:
            if position == colorH: 
              if position == colorH2: state = 244
              else: state = 245
            else: 
              if position == colorH2: state = 246
              else: state = 247
        else:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 248
              else: state = 249
            else: 
              if position == colorH2: state = 250
              else: state = 251
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 252
              else: state = 253
            else: 
              if position == colorH2: state = 254
              else: state = 255
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 256
              else: state = 257
            else: 
              if position == colorH2: state = 258
              else: state = 259
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 260
              else: state = 261
            else: 
              if position == colorH2: state = 262
              else: state = 263
          else:
            if position == colorH: 
              if position == colorH2: state = 264
              else: state = 265
            else: 
              if position == colorH2: state = 266
              else: state = 267
  elif position == -1:
    if color == 0:
      if reward >= 0:
        if n_hold < 288:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 268
              else: state = 269
            else: 
              if position == colorH2: state = 270
              else: state = 271
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 272
              else: state = 273
            else: 
              if position == colorH2: state = 274
              else: state = 275
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 276
              else: state = 277
            else: 
              if position == colorH2: state = 278
              else: state = 279
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 280
              else: state = 281
            else: 
              if position == colorH2: state = 282
              else: state = 283
          else:
            if position == colorH: 
              if position == colorH2: state = 284
              else: state = 285
            else: 
              if position == colorH2: state = 286
              else: state = 287
        else:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 288
              else: state = 289
            else: 
              if position == colorH2: state = 290
              else: state = 291
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 292
              else: state = 293
            else: 
              if position == colorH2: state = 294
              else: state = 295
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 296
              else: state = 297
            else: 
              if position == colorH2: state = 298
              else: state = 299
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 300
              else: state = 301
            else: 
              if position == colorH2: state = 302
              else: state = 303
          else:
            if position == colorH: 
              if position == colorH2: state = 304
              else: state = 305
            else: 
              if position == colorH2: state = 306
              else: state = 307
      elif reward < 0:
        if n_hold < 288:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 308
              else: state = 309
            else: 
              if position == colorH2: state = 310
              else: state = 311
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 312
              else: state = 313
            else: 
              if position == colorH2: state = 314
              else: state = 315
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 316
              else: state = 317
            else: 
              if position == colorH2: state = 318
              else: state = 319
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 320
              else: state = 321
            else: 
              if position == colorH2: state = 322
              else: state = 323
          else:
            if position == colorH: 
              if position == colorH2: state = 324
              else: state = 325
            else: 
              if position == colorH2: state = 326
              else: state = 327
        else:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 328
              else: state = 329
            else: 
              if position == colorH2: state = 330
              else: state = 331
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 332
              else: state = 333
            else: 
              if position == colorH2: state = 334
              else: state = 335
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 336
              else: state = 337
            else: 
              if position == colorH2: state = 338
              else: state = 339
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 340
              else: state = 341
            else: 
              if position == colorH2: state = 342
              else: state = 343
          else:
            if position == colorH: 
              if position == colorH2: state = 344
              else: state = 345
            else: 
              if position == colorH2: state = 346
              else: state = 347
    elif color == 1:
      if reward >= 0:
        if n_hold < 288:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 348
              else: state = 349
            else: 
              if position == colorH2: state = 350
              else: state = 351
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 352
              else: state = 353
            else: 
              if position == colorH2: state = 354
              else: state = 355
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 356
              else: state = 357
            else: 
              if position == colorH2: state = 358
              else: state = 359
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 360
              else: state = 361
            else: 
              if position == colorH2: state = 362
              else: state = 363
          else:
            if position == colorH: 
              if position == colorH2: state = 364
              else: state = 365
            else: 
              if position == colorH2: state = 366
              else: state = 367
        else:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 368
              else: state = 369
            else: 
              if position == colorH2: state = 370
              else: state = 371
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 372
              else: state = 373
            else: 
              if position == colorH2: state = 374
              else: state = 375
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 376
              else: state = 377
            else: 
              if position == colorH2: state = 378
              else: state = 379
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 380
              else: state = 381
            else: 
              if position == colorH2: state = 382
              else: state = 383
          else:
            if position == colorH: 
              if position == colorH2: state = 384
              else: state = 385
            else: 
              if position == colorH2: state = 386
              else: state = 387
      elif reward < 0:
        if n_hold < 288:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 388
              else: state = 389
            else: 
              if position == colorH2: state = 390
              else: state = 391
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 392
              else: state = 393
            else: 
              if position == colorH2: state = 394
              else: state = 395
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 396
              else: state = 397
            else: 
              if position == colorH2: state = 398
              else: state = 399
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 400
              else: state = 401
            else: 
              if position == colorH2: state = 402
              else: state = 403
          else:
            if position == colorH: 
              if position == colorH2: state = 404
              else: state = 405
            else: 
              if position == colorH2: state = 406
              else: state = 407
        else:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 408
              else: state = 409
            else: 
              if position == colorH2: state = 410
              else: state = 411
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 412
              else: state = 413
            else: 
              if position == colorH2: state = 414
              else: state = 415
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 416
              else: state = 417
            else: 
              if position == colorH2: state = 418
              else: state = 419
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 420
              else: state = 421
            else: 
              if position == colorH2: state = 422
              else: state = 423
          else:
            if position == colorH: 
              if position == colorH2: state = 424
              else: state = 425
            else: 
              if position == colorH2: state = 426
              else: state = 427
    elif color == -1:
      if reward >= 0:
        if n_hold < 288:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 428
              else: state = 429
            else: 
              if position == colorH2: state = 430
              else: state = 431
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 432
              else: state = 433
            else: 
              if position == colorH2: state = 434
              else: state = 435
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 436
              else: state = 437
            else: 
              if position == colorH2: state = 438
              else: state = 439
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 440
              else: state = 441
            else: 
              if position == colorH2: state = 442
              else: state = 443
          else:
            if position == colorH: 
              if position == colorH2: state = 444
              else: state = 445
            else: 
              if position == colorH2: state = 446
              else: state = 447
        else:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 448
              else: state = 449
            else: 
              if position == colorH2: state = 450
              else: state = 451
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 452
              else: state = 453
            else: 
              if position == colorH2: state = 454
              else: state = 455
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 456
              else: state = 457
            else: 
              if position == colorH2: state = 458
              else: state = 459
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 460
              else: state = 461
            else: 
              if position == colorH2: state = 462
              else: state = 463
          else:
            if position == colorH: 
              if position == colorH2: state = 464
              else: state = 465
            else: 
              if position == colorH2: state = 466
              else: state = 467
      elif reward < 0:
        if n_hold < 288:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 468
              else: state = 469
            else: 
              if position == colorH2: state = 470
              else: state = 471
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 472
              else: state = 473
            else: 
              if position == colorH2: state = 474
              else: state = 475
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 476
              else: state = 477
            else: 
              if position == colorH2: state = 478
              else: state = 479
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 480
              else: state = 481
            else: 
              if position == colorH2: state = 482
              else: state = 483
          else:
            if position == colorH: 
              if position == colorH2: state = 484
              else: state = 485
            else: 
              if position == colorH2: state = 486
              else: state = 487
        else:
          if profit <= 3:
            if position == colorH: 
              if position == colorH2: state = 488
              else: state = 489
            else: 
              if position == colorH2: state = 490
              else: state = 491
          elif profit > 3 and profit <= 10:
            if position == colorH: 
              if position == colorH2: state = 492
              else: state = 493
            else: 
              if position == colorH2: state = 494
              else: state = 495
          elif profit > 10 and profit <= 17:
            if position == colorH: 
              if position == colorH2: state = 496
              else: state = 497
            else: 
              if position == colorH2: state = 498
              else: state = 499
          elif profit > 17 and profit <= 27:
            if position == colorH: 
              if position == colorH2: state = 500
              else: state = 501
            else: 
              if position == colorH2: state = 502
              else: state = 503
          else:
            if position == colorH: 
              if position == colorH2: state = 504
              else: state = 505
            else: 
              if position == colorH2: state = 506
              else: state = 507
  return state

def q_state13(position, reward, profit, n_hold, n_delta, color, colorH, colorH2):
  state = 0
  if position == 0:
    if color == 0:
      if reward == 0:
        if n_delta <= 5:
          if position == colorH: 
            if position == colorH2: state = 1
            else: state = 2
          else: 
            if position == colorH2: state = 3
            else: state = 4
        elif n_delta > 5 and n_delta <=12:
          if position == colorH: 
            if position == colorH2: state = 5
            else: state = 6
          else: 
            if position == colorH2: state = 7
            else: state = 8
        elif n_delta > 12 and n_delta <= 48:
          if position == colorH: 
            if position == colorH2: state = 9
            else: state = 10
          else: 
            if position == colorH2: state = 11
            else: state = 12
        else: 
          if position == colorH: 
            if position == colorH2: state = 13
            else: state = 14
          else: 
            if position == colorH2: state = 15
            else: state = 16
      else:
        state = 17
    elif color == 1:
      if reward == 0:
        if n_delta <= 5:
          if position == colorH: 
            if position == colorH2: state = 18
            else: state = 19
          else: 
            if position == colorH2: state = 20
            else: state = 21
        elif n_delta > 5 and n_delta<=12:
          if position == colorH: 
            if position == colorH2: state = 22
            else: state = 23
          else: 
            if position == colorH2: state = 24
            else: state = 25
        elif n_delta > 12 and n_delta<= 48:
          if position == colorH: 
            if position == colorH2: state = 26
            else: state = 27
          else: 
            if position == colorH2: state = 28
            else: state = 29
        else: 
          if position == colorH: 
            if position == colorH2: state = 30
            else: state = 31
          else: 
            if position == colorH2: state = 32
            else: state = 33
      else:
        state = 34
    elif color == -1:
      if reward == 0:
        if n_delta <= 5:
          if position == colorH: 
            if position == colorH2: state = 35
            else: state = 36
          else: 
            if position == colorH2: state = 37
            else: state = 38
        elif n_delta > 5 and n_delta<=12:
          if position == colorH: 
            if position == colorH2: state = 39
            else: state = 40
          else: 
            if position == colorH2: state = 41
            else: state = 42
        elif n_delta > 12 and n_delta<= 48:
          if position == colorH: 
            if position == colorH2: state = 43
            else: state = 44
          else: 
            if position == colorH2: state = 45
            else: state = 46
        else: 
          if position == colorH: 
            if position == colorH2: state = 47
            else: state = 48
          else: 
            if position == colorH2: state = 49
            else: state = 50
      else:
        state = 51
  elif position == 1:
    if color == 0:
      if reward >= 0:
        if n_hold <= 5:
          if position == colorH: 
            if position == colorH2: state = 52
            else: state = 53
          else: 
            if position == colorH2: state = 54
            else: state = 55
        elif n_hold > 5 and n_hold<=12:
          if position == colorH: 
            if position == colorH2: state = 56
            else: state = 57
          else: 
            if position == colorH2: state = 58
            else: state = 59
        elif n_hold > 12 and n_hold<= 48:
          if position == colorH: 
            if position == colorH2: state = 60
            else: state = 61
          else: 
            if position == colorH2: state = 62
            else: state = 63
        else: 
          if position == colorH: 
            if position == colorH2: state = 64
            else: state = 65
          else: 
            if position == colorH2: state = 66
            else: state = 67
      elif reward < 0:
        if n_hold <= 5:
          if position == colorH: 
            if position == colorH2: state = 68
            else: state = 69
          else: 
            if position == colorH2: state = 70
            else: state = 71
        elif n_hold > 5 and n_hold <=12:
          if position == colorH: 
            if position == colorH2: state = 72
            else: state = 73
          else: 
            if position == colorH2: state = 74
            else: state = 75
        elif n_hold > 12 and n_hold<= 48:
          if position == colorH: 
            if position == colorH2: state = 76
            else: state = 77
          else: 
            if position == colorH2: state = 78
            else: state = 79
        else: 
          if position == colorH: 
            if position == colorH2: state = 80
            else: state = 81
          else: 
            if position == colorH2: state = 82
            else: state = 83
    elif color == 1:
      if reward >= 0:
        if n_hold <= 5:
          if position == colorH: 
            if position == colorH2: state = 84
            else: state = 85
          else: 
            if position == colorH2: state = 86
            else: state = 87
        elif n_hold > 5 and n_hold<=12:
          if position == colorH: 
            if position == colorH2: state = 88
            else: state = 89
          else: 
            if position == colorH2: state = 90
            else: state = 91
        elif n_hold > 12 and n_hold<=48:
          if position == colorH: 
            if position == colorH2: state = 92
            else: state = 93
          else: 
            if position == colorH2: state = 94
            else: state = 95
        else: 
          if position == colorH: 
            if position == colorH2: state = 96
            else: state = 97
          else: 
            if position == colorH2: state = 98
            else: state = 99
      elif reward < 0:
        if n_hold <= 5:
          if position == colorH: 
            if position == colorH2: state = 100
            else: state = 101
          else: 
            if position == colorH2: state = 102
            else: state = 103
        elif n_hold > 5 and n_hold<=12:
          if position == colorH: 
            if position == colorH2: state = 104
            else: state = 105
          else: 
            if position == colorH2: state = 106
            else: state = 107
        elif n_hold > 12 and n_hold<= 48:
          if position == colorH: 
            if position == colorH2: state = 108
            else: state = 109
          else: 
            if position == colorH2: state = 110
            else: state = 111
        else: 
          if position == colorH: 
            if position == colorH2: state = 112
            else: state = 113
          else: 
            if position == colorH2: state = 114
            else: state = 115
    elif color == -1:
      if reward >= 0:
        if n_hold <= 5:
          if position == colorH: 
            if position == colorH2: state = 116
            else: state = 117
          else: 
            if position == colorH2: state = 118
            else: state = 119
        elif n_hold > 5 and n_hold<=12:
          if position == colorH: 
            if position == colorH2: state = 120
            else: state = 121
          else: 
            if position == colorH2: state = 122
            else: state = 123
        elif n_hold > 12 and n_hold<= 48:
          if position == colorH: 
            if position == colorH2: state = 124
            else: state = 125
          else: 
            if position == colorH2: state = 126
            else: state = 127
        else: 
          if position == colorH: 
            if position == colorH2: state = 128
            else: state = 129
          else: 
            if position == colorH2: state = 130
            else: state = 131
      elif reward < 0:
        if n_hold <= 5:
          if position == colorH: 
            if position == colorH2: state = 132
            else: state = 133
          else: 
            if position == colorH2: state = 134
            else: state = 135
        elif n_hold > 5 and n_hold<=12:
          if position == colorH: 
            if position == colorH2: state = 136
            else: state = 137
          else: 
            if position == colorH2: state = 138
            else: state = 139
        elif n_hold > 12 and n_hold<= 48:
          if position == colorH: 
            if position == colorH2: state = 140
            else: state = 141
          else: 
            if position == colorH2: state = 142
            else: state = 143
        else: 
          if position == colorH: 
            if position == colorH2: state = 144
            else: state = 145
          else: 
            if position == colorH2: state = 146
            else: state = 147
  elif position == -1:
    if color == 0:
      if reward >= 0:
        if n_hold <= 5:
          if position == colorH: 
            if position == colorH2: state = 148
            else: state = 149
          else: 
            if position == colorH2: state = 150
            else: state = 151
        elif n_hold > 5 and n_hold<=12:
          if position == colorH: 
            if position == colorH2: state = 152
            else: state = 153
          else: 
            if position == colorH2: state = 154
            else: state = 155
        elif n_hold > 12 and n_hold<= 48:
          if position == colorH: 
            if position == colorH2: state = 156
            else: state = 157
          else: 
            if position == colorH2: state = 158
            else: state = 159
        else: 
          if position == colorH: 
            if position == colorH2: state = 160
            else: state = 161
          else: 
            if position == colorH2: state = 162
            else: state = 163
      elif reward < 0:
        if n_hold <= 5:
          if position == colorH: 
            if position == colorH2: state = 164
            else: state = 165
          else: 
            if position == colorH2: state = 166
            else: state = 167
        elif n_hold > 5 and n_hold<=12:
          if position == colorH: 
            if position == colorH2: state = 168
            else: state = 169
          else: 
            if position == colorH2: state = 170
            else: state = 171
        elif n_hold > 12 and n_hold<= 48:
          if position == colorH: 
            if position == colorH2: state = 172
            else: state = 173
          else: 
            if position == colorH2: state = 174
            else: state = 175
        else: 
          if position == colorH: 
            if position == colorH2: state = 176
            else: state = 177
          else: 
            if position == colorH2: state = 178
            else: state = 179
    elif color == 1:
      if reward >= 0:
        if n_hold <= 5:
          if position == colorH: 
            if position == colorH2: state = 180
            else: state = 181
          else: 
            if position == colorH2: state = 182
            else: state = 183
        elif n_hold > 5 and n_hold<=12:
          if position == colorH: 
            if position == colorH2: state = 184
            else: state = 185
          else: 
            if position == colorH2: state = 186
            else: state = 187
        elif n_hold > 12 and n_hold<= 48:
          if position == colorH: 
            if position == colorH2: state = 188
            else: state = 189
          else: 
            if position == colorH2: state = 190
            else: state = 191
        else: 
          if position == colorH: 
            if position == colorH2: state = 192
            else: state = 193
          else: 
            if position == colorH2: state = 194
            else: state = 195
      elif reward < 0:
        if n_hold <= 5:
          if position == colorH: 
            if position == colorH2: state = 196
            else: state = 197
          else: 
            if position == colorH2: state = 198
            else: state = 199
        elif n_hold > 5 and n_hold<=12:
          if position == colorH: 
            if position == colorH2: state = 200
            else: state = 201
          else: 
            if position == colorH2: state = 202
            else: state = 203
        elif n_hold > 12 and n_hold<= 48:
          if position == colorH: 
            if position == colorH2: state = 204
            else: state = 205
          else: 
            if position == colorH2: state = 206
            else: state = 207
        else: 
          if position == colorH: 
            if position == colorH2: state = 208
            else: state = 209
          else: 
            if position == colorH2: state = 210
            else: state = 211
    elif color == -1:
      if reward >= 0:
        if n_hold <= 5:
          if position == colorH: 
            if position == colorH2: state = 212
            else: state = 213
          else: 
            if position == colorH2: state = 214
            else: state = 215
        elif n_hold > 5 and n_hold<=12:
          if position == colorH: 
            if position == colorH2: state = 216
            else: state = 217
          else: 
            if position == colorH2: state = 218
            else: state = 219
        elif n_hold > 12 and n_hold<= 48:
          if position == colorH: 
            if position == colorH2: state = 220
            else: state = 221
          else: 
            if position == colorH2: state = 222
            else: state = 223
        else: 
          if position == colorH: 
            if position == colorH2: state = 224
            else: state = 225
          else: 
            if position == colorH2: state = 226
            else: state = 227
      elif reward < 0:
        if n_hold <= 5:
          if position == colorH: 
            if position == colorH2: state = 228
            else: state = 229
          else: 
            if position == colorH2: state = 230
            else: state = 231
        elif n_hold > 5 and n_hold<=12:
          if position == colorH: 
            if position == colorH2: state = 232
            else: state = 233
          else: 
            if position == colorH2: state = 234
            else: state = 235
        elif n_hold > 12 and n_hold<= 48:
          if position == colorH: 
            if position == colorH2: state = 236
            else: state = 237
          else: 
            if position == colorH2: state = 238
            else: state = 239
        else: 
          if position == colorH: 
            if position == colorH2: state = 240
            else: state = 241
          else: 
            if position == colorH2: state = 242
            else: state = 243
  return state


def q_state15(position, reward, n_hold, n_delta, color, colorH, colorH2):
  state = 0
  if position == 0:
    if color == 0:
      if reward == 0:
        if position == colorH: 
            if position == colorH2: state = 1
            else: state = 2
        else: 
            if position == colorH2: state = 3
            else: state = 4
      else:
        if position == colorH: 
            if position == colorH2: state = 5
            else: state = 6
        else: 
            if position == colorH2: state = 7
            else: state = 8
    elif color == 1:
      if reward == 0:
        if position == colorH: 
            if position == colorH2: state = 9
            else: state = 10
        else: 
            if position == colorH2: state = 11
            else: state = 12
      else:
        if position == colorH: 
            if position == colorH2: state = 13
            else: state = 14
        else: 
            if position == colorH2: state = 15
            else: state = 16
    elif color == -1:
      if reward == 0:
        if position == colorH: 
            if position == colorH2: state = 17
            else: state = 18
        else: 
            if position == colorH2: state = 19
            else: state = 20
      else:
        if position == colorH: 
            if position == colorH2: state = 21
            else: state = 22
        else: 
            if position == colorH2: state = 23
            else: state = 24
  elif position == 1:
    if color == 0:
      if reward >= 0:
        if position == colorH: 
            if position == colorH2: state = 25
            else: state = 26
        else: 
            if position == colorH2: state = 27
            else: state = 28
      else:
        if position == colorH: 
            if position == colorH2: state = 29
            else: state = 30
        else: 
            if position == colorH2: state = 31
            else: state = 32
    elif color == 1:
      if reward >= 0:
        if position == colorH: 
            if position == colorH2: state = 33
            else: state = 34
        else: 
            if position == colorH2: state = 35
            else: state = 36
      else:
        if position == colorH: 
            if position == colorH2: state = 37
            else: state = 38
        else: 
            if position == colorH2: state = 39
            else: state = 40
    elif color == -1:
      if reward >= 0:
        if position == colorH: 
            if position == colorH2: state = 41
            else: state = 42
        else: 
            if position == colorH2: state = 43
            else: state = 44
      else:
        if position == colorH: 
            if position == colorH2: state = 45
            else: state = 46
        else: 
            if position == colorH2: state = 47
            else: state = 48
  elif position == -1:
    if color == 0:
      if reward >= 0:
        if position == colorH: 
            if position == colorH2: state = 49
            else: state = 50
        else: 
            if position == colorH2: state = 51
            else: state = 52
      else:
          if position == colorH: 
            if position == colorH2: state = 53
            else: state = 54
          else: 
            if position == colorH2: state = 55
            else: state = 56
    elif color == 1:
      if reward >= 0:
        if position == colorH: 
            if position == colorH2: state = 57
            else: state = 58
        else: 
            if position == colorH2: state = 59
            else: state = 60
      else:
        if position == colorH: 
            if position == colorH2: state = 61
            else: state = 62
        else: 
            if position == colorH2: state = 63
            else: state = 64
    elif color == -1:
      if reward >= 0:
        if position == colorH: 
            if position == colorH2: state = 65
            else: state = 66
        else: 
            if position == colorH2: state = 67
            else: state = 68
      else:
        if position == colorH: 
            if position == colorH2: state = 69
            else: state = 70
        else: 
            if position == colorH2: state = 71
            else: state = 72
  return state

def agent_lstm(policy, obs):
  output = policy.predict(obs)
  act = output[-1]#action.numpy() #
  max_index = np.where(act == np.amax(act))
		#print(max_index[0])
  if max_index[0] == [0]: action_name = HOLD
  elif max_index[0] == [1]: action_name = BUY
  elif max_index[0] == [2]: action_name = SELL
  return action_name

def relu(x):
  if x > 0:
    return x
  else:
    return 0

def action(q_arr, policy, obs):
  max_index = np.where(q_arr == np.amax(q_arr))
  if len(max_index[0]) >1: action_q_name = agent_lstm(policy, obs)
  elif max_index[0] == [0]: action_q_name = HOLD
  elif max_index[0] == [1]: action_q_name = BUY
  elif max_index[0] == [2]: action_q_name = SELL
  return action_q_name

def fillS(state, arr):
  x = np.where(np.array(arr)==state)
  if len(x[0]) == 0:
    arr.append(state)
  return arr
    
def test_q(path):
  env = Environment3_9_4h1h_revised_test2_limit.Env(path)
  Q = pickle.load(open('./Q_2008-2016_2_env39r_state15_minmax_4h1h_pabs.pkl', 'rb'))
  arr_s = []
  arr_s_temp = []
  policy = tf.keras.models.load_model("D:/TESIS HANA/code thesis/lstm_exp/bener/1training1layer_20_16_64unit_2008-2016_barupj_1steplabel_color3pips.h5")
  t = 47
  obs = env.reset()
  q_arr = Q[0]
  act = action(q_arr, policy, obs)
  obs, position, rew, dd,profit, holding_time, delta_time, color,color1h,color4h,done,_ = env.step(act)
  state = q_state15(position, rew,holding_time, delta_time, color,color1h,color4h)
  #arr_s = fillS(state, arr_s)
  done = False
  while t <= env.df.shape[0]-1:
		#print(action_name)
    q_arr = Q[state,:]
    action_q_name = action(q_arr, policy, obs)
    obs,position, next_rew,dd, profit, holding_time, delta_time, color,color1h,color4h,done,_ = env.step(action_q_name)
    print('position: ',position)
    print('color: ',color)
    print('color1h: ',color1h)
    print('color4h: ',color4h)
    nextState = q_state15(position,next_rew,holding_time, delta_time, color,color1h,color4h)
    print('state: ',nextState)
    print('drawdown: ',dd)
    print('profit: ',profit)
    print('reward: ',next_rew)
    #arr_s = fillS(nextState, arr_s)
    state = nextState
    env.save_array()
    if profit > 0 and holding_time > 0:
      arr_s_temp.append(state)
    if len(arr_s_temp) > 0 and holding_time == 0:
      arr_s.append(arr_s_temp)
      arr_s_temp = []
    pickle.dump(arr_s, open('./state15_profit_2022.pkl', 'wb'))
    if state == 0:
      break
    if done == True:
      break
    t += 1
    #print(arr_s)

PATH_TEST = "D:/TESIS HANA/data/alpari/testing/EURUSD_5M_2022processed_lstm_barupj_Daily_bener.csv"
#PATH_ACTOR = "D:/TESIS HANA/code thesis/ppo_from_scratch/ppo_sharedactor.pth"

test_q(PATH_TEST)	

