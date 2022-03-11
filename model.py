#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
scenarios = ["s0","s1","s2","s3","s4","s5"]
pd.options.display.max_columns = 500
pd.options.display.max_rows = 200
pd.set_option('display.float_format', lambda x: '%.2f' % x)
import gurobipy as gp
from gurobipy import GRB
import pickle
import numpy_financial as npf


def model_pro(db, num, p, sell):
    dt = pickle.load(open(r'C:\Users\sreza\Google Drive\nb\sum20\0625\results\date.pkl', 'rb'))
    # getting parameters
    tk = [-1, 2975, 5664, 8635, 11515, 14491, 17371, 20347, 23323, 26203, 29179, 32063, 34943]
    K = 12
    E_max = db['E_max']
    E_min = db['E_min']
    B_max = db['B_max']
    B_min = db['B_min']
    T = db['T']
    U_c = db['U_c']
    U_d = db['U_d']
    eta_c = db['eta_c']
    eta_d = db['eta_d']
    R = db['R']
    L = db['L']
    A = db['A']
    V = db['V']
    Pen = db['Pen']
    if p == 'flat':
        P_CHARGE = db['P_fx_charge']
        P = db['P_fx']
    elif p == 'Time-of-Use':
        P_CHARGE = db['P_tou_charge']
        P = db['P_tou']
    elif p == 'RTLMP':
        P_CHARGE = db['P_tou_charge']*0
        P = db['P_gmp']
    elif p == 'TOU&RTLMP':
        P_CHARGE = db['P_tou_charge']
        P = db['P_tou']
    else:
        P_CHARGE = db['P_fx_charge']
        P = db['P_fx']
        P_EV = db['P_ev']
    if sell == True:
        if p == 'TOU&RTLMP':
            p_sell = db['P_gmp']
        else:
            p_sell = P
    else :
        p_sell = P*0
        
    # Model formulation
    m = gp.Model('gmp')
    # Decision Vars
    x_GL   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^GL')
    x_GB   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^GB')
    x_GE   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^GE')
    x_RL   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^RL')
    x_RB   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^RB')
    x_RE   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^RE')
    x_RG   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^RG')
    x_BL   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^BL')
    x_EL   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^EL')
    x_BG   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^BG')
    x_EG   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^EG')
    x_loss = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^loss')
    b_E    = m.addVars(T, vtype=GRB.CONTINUOUS, name='b^E')
    b_B    = m.addVars(T, vtype=GRB.CONTINUOUS, name='b^B')
    # Constraints
    m.addConstrs((x_GL[t] + eta_d * x_BL[t] + eta_d * x_EL[t] + x_RL[t] + x_loss[t] == L[t] for t in range(T)), name='c_load_2')
    m.addConstrs((x_BL[t] + x_BG[t] <= b_B[t] for t in range(T)), name='c_bat_state_2')
    m.addConstrs((x_EL[t] + x_EG[t] <= b_E[t] * A[t] for t in range(T)), name='c_ev_state_2')
    m.addConstrs((x_RE[t] + x_GE[t] <= E_max * A[t] for t in range(T)), name='c_ev_2_state_2')
    m.addConstrs((x_BL[t] + x_BG[t] + x_EL[t] + x_EG[t] <= U_d for t in range(T)), name='c_dch_max_2')
    m.addConstrs((x_GB[t] + x_RB[t] + x_GE[t] + x_RE[t] <= U_c for t in range(T)), name='c_ch_max_2')
    m.addConstrs((b_B[t] <= B_max for t in range(T)), name='c_b_B_max_2')
    m.addConstrs((b_B[t] >= B_min for t in range(T)), name='c_b_B_min_2')
    m.addConstrs((b_E[t] <= E_max for t in range(T)), name='c_b_E_max_2')
    m.addConstrs((b_E[t] >= E_min for t in range(T)), name='c_b_E_min_2')
    m.addConstrs((x_RL[t] + x_RB[t] + x_RE[t] + x_RG[t] <= R[t] for t in range(T)), name='c_gen_cap_2')
    m.addConstrs((b_B[t] + eta_c * (x_GB[t] + x_RB[t]) - (x_BL[t] + x_BG[t]) == b_B[t+1] for t in range(T-1)), name='c_next_state_b_B_2')
    m.addConstrs((b_E[t] + eta_c * (x_GE[t] + x_RE[t]) - (x_EL[t] + x_EG[t]) - (V[t] / eta_d) == b_E[t+1] for t in range(T-1)), name='c_next_state_b_E_2')
    if sell == False:
        m.addConstrs((x_BG[t] + x_RG[t] + x_EG[t] == 0 for t in range(T)), name='c_trade')
    # Objective
    if p == 'Time-of-Use EV & flat':
        m.setObjective(gp.quicksum(P_CHARGE + P_EV[t] * 1.0001 * x_GE[t] + P[t] * 1.0001 * (x_GB[t] + x_GL[t]) - 0.9999 * p_sell[t] * (x_RG[t] + eta_d * (x_BG[t] + x_EG[t])) + Pen * x_loss[t] for t in range(T)), GRB.MINIMIZE)
    else:
        m.setObjective(gp.quicksum(P_CHARGE + P[t] * 1.0001 * (x_GB[t] + x_GE[t] + x_GL[t]) - p_sell[t] * 0.999 * (x_RG[t] + eta_d * (x_BG[t] + x_EG[t])) + Pen * x_loss[t] for t in range(T)), GRB.MINIMIZE)
    # Optimize
    m.optimize()
    # result
#     m.write("gmp-output_%g.sol" % num)
    columns = "x^GL, x^GB, x^GE, x^RL, x^RB, x^RE, x^RG, x^BL, x^EL, x^BL * eta_d, x^EL * eta_d, x^BG, x^EG, x^BG * eta_d, x^EG * eta_d, b^B, b^E, x^loss, Peak load, V, P (GMP), P (Flat), P (ToU), P (ToU for EV), R, L, Default cost of load covering at flat rate, Default cost of load covering at Time-of-Use rate, Capacity cost, Cost of load covering by GMP, GMP's cost, Pro's cost, Loss cost, EV's cost, Agent, Rate, Sell".split(', ')
    result = pd.DataFrame(columns=columns, index=range(T), data=0.0)
    GL = np.zeros(T)
    GB = np.zeros(T)
    GE = np.zeros(T)
    RL = np.zeros(T)
    RB = np.zeros(T)
    RE = np.zeros(T)
    RG = np.zeros(T)
    BL = np.zeros(T)
    EL = np.zeros(T)
    BG = np.zeros(T)
    EG = np.zeros(T)
    ls = np.zeros(T)
    xb_B = np.zeros(T)
    xb_E = np.zeros(T)
    peak = np.zeros(T)
    l_fx_cost = np.zeros(T)
    l_tou_cost = np.zeros(T)
    cap_cost = np.zeros(T)
    l_gmp_cost = np.zeros(T)
    gmp_cost = np.zeros(T)
    pro_cost = np.zeros(T)
    ev_cost = np.zeros(T)
    ls_cost = np.zeros(T)

    for k in range(K):
        peak[tk[k]+1:tk[k+1]+1] = 4 * np.max([x_GB[t].x +  x_GE[t].x + x_GL[t].x for t in range(tk[k]+1,tk[k+1])])
        cap_cost[tk[k]+1:tk[k+1]+1] = (1.35*5+9) * peak[tk[k]+1:tk[k+1]+1]
    
    for t in range(T):
        GL[t] = x_GL[t].x
        GB[t] = x_GB[t].x
        GE[t] = x_GE[t].x
        RL[t] = x_RL[t].x
        RB[t] = x_RB[t].x
        RE[t] = x_RE[t].x
        RG[t] = x_RG[t].x
        BL[t] = x_BL[t].x
        EL[t] = x_EL[t].x
        BG[t] = x_BG[t].x
        EG[t] = x_EG[t].x
        ls[t] = x_loss[t].x
        xb_B[t] = b_B[t].x
        xb_E[t] = b_E[t].x
        l_fx_cost[t] = db['P_fx_charge'] + L[t] * db['P_fx'][t]
        l_tou_cost[t] = db['P_tou_charge'] + L[t] * db['P_tou'][t]
        l_gmp_cost[t] = db['P_gmp'][t] * (GB[t] + GE[t] + GL[t] - RG[t] - db['eta_d'] * (BG[t] + EG[t]))
        if p == 'Time-of-Use EV & flat':
            pro_cost[t] = db['P_fx_charge'] + (GB[t]  + GL[t]) * P[t] - (RG[t] + db['eta_d']*(BG[t] + EG[t])) * p_sell[t] + GE[t] * db['P_ev'][t]
        else:
            pro_cost[t] = P_CHARGE + (GB[t] + GE[t] + GL[t]) * P[t] - (RG[t] + db['eta_d'] * (BG[t] + EG[t])) * p_sell[t]
        ev_cost[t] = P[t] * V[t] / eta_c/ eta_d
        ls_cost[t] = Pen * ls[t]
     

    result.loc[:,'Sell'] = sell
    result.loc[:,'Agent'] = 'PRO'
    result.loc[:,'Rate'] = p
    result.loc[:,'x^GL'] = GL
    result.loc[:,'x^GB'] = GB
    result.loc[:,'x^GE'] = GE
    result.loc[:,'x^RL'] = RL
    result.loc[:,'x^RB'] = RB
    result.loc[:,'x^RE'] = RE
    result.loc[:,'x^RG'] = RG
    result.loc[:,'x^BL'] = BL
    result.loc[:,'x^EL'] = EL
    result.loc[:,'x^BG'] = BG
    result.loc[:,'x^EG'] = EG
    result.loc[:,'x^BL * eta_d'] = BL * db['eta_d']
    result.loc[:,'x^BG * eta_d'] = BG * db['eta_d']
    result.loc[:,'x^EL * eta_d'] = EL * db['eta_d']
    result.loc[:,'x^EG * eta_d'] = EG * db['eta_d']
    result.loc[:,'x^loss'] = ls
    result.loc[:,'Peak load'] = peak
    result.loc[:,'V'] = V
    result.loc[:,'b^B'] = xb_B
    result.loc[:,'b^E'] = xb_E
    result.loc[:,'P (GMP)'] = db['P_gmp']
    result.loc[:,'P (Flat)'] = db['P_fx']
    result.loc[:,'P (ToU)'] = db['P_tou']
    result.loc[:,"P (ToU for EV)"] = db['P_ev']
    result.loc[:,'R'] = R
    result.loc[:,'L'] = L
    result.loc[:,'Default cost of load covering at flat rate'] = l_fx_cost
    result.loc[:,'Default cost of load covering at Time-of-Use rate'] = l_tou_cost
    result.loc[:,'Capacity cost'] = cap_cost
    result.loc[:,'Cost of load covering by GMP'] = l_gmp_cost
    result.loc[:,"Pro's cost"] = pro_cost
    result.loc[:,"EV's cost"] = ev_cost
    result.loc[:,'Loss cost'] = ls_cost
           
    result = pd.concat([dt, result], axis = 1)

    
    return result

def pro_ctrl(run,house_num):
    t = [0, 2977, 5665, 8637, 11517, 14493, 17373, 20349, 23325, 26205, 29181, 32065, 34945]
    index = 'Jan.Feb.Mar.Apr.May.Jun.Jul.Aug.Sep.Oct.Nov.Dec'.split('.')
    col2 = "x^GL, x^GB, x^GE, x^RL, x^RB, x^RE, x^RG, x^BL, x^EL, x^BL * eta_d, x^EL * eta_d, x^BG, x^EG, x^BG * eta_d, x^EG * eta_d, b^B, b^E, x^loss, Peak load, V, P (GMP), P (Flat), P (ToU), P (ToU for EV), R, L, Default cost of load covering at flat rate, Default cost of load covering at Time-of-Use rate, Capacity cost, Cost of load covering by GMP, GMP's cost, Pro's cost, Loss cost, EV's cost, Agent, Rate, Sell".split(', ')


    N= 15
    ii = 0.0619



    d1 = 61
    d2 = d1+6
    T1 = 24*4*(d1-1)
    T2 = 24*4*d2

    scenarios = {
      "s0" : {'B_max': 0,    'B_min': 0,  'U_c': 0,    'U_d': 0,    'E_max': 0,  'E_min': 0,  'use_pv': False, 'price': 0},
      "s1" : {'B_max': 40.5, 'B_min': 0,  'U_c': 3.75, 'U_d': 3.75, 'E_max': 0,  'E_min': 0,  'use_pv': False, 'price': 27300},
      "s2" : {'B_max': 27,   'B_min': 0,  'U_c': 2.5,  'U_d': 2.5,  'E_max': 0,  'E_min': 0,  'use_pv': True,  'price': 35125},
      "s3" : {'B_max': 0,    'B_min': 0,  'U_c': 1.9,  'U_d': 1.9,  'E_max': 60, 'E_min': 24, 'use_pv': False, 'price': 4000},
      "s4" : {'B_max': 0,    'B_min': 0,  'U_c': 1.9,  'U_d': 1.9,  'E_max': 60, 'E_min': 24, 'use_pv': True,  'price': 17145},
      "s5" : {'B_max': 10,   'B_min': 0,  'U_c': 1.9,  'U_d': 1.9,  'E_max': 60, 'E_min': 24, 'use_pv': True,  'price': 23724}
    }
    mark = 0
    for s in scenarios:
        use_pv = scenarios[s]['use_pv']
        bat = 'Always'
        db = pickle.load(open(rf'C:\Users\sreza\Google Drive\nb\sum20\0625\results\db_h_{house_num}_RES_{use_pv}_ESS_{bat}.pkl', 'rb'))
        h_st = 8
        m_st = 45
        h_fn = 17
        m_fn = 15
        usage = 10
        for p in ['flat', 'Time-of-Use', 'Time-of-Use EV & flat', 'RTLMP', 'TOU&RTLMP']:
            for sell in [False, True]:
                for i in scenarios[s]:
                    db[i] = scenarios[s][i]

                db['P_ev'] = db['P'].copy()
                db['P_ev'][:] = 0.12831
                db['P_gmp'] = db['P'].copy()
                db['P_fx'] = db['P'].copy()
                db['P_fx'][:] = 0.16859
                db['P_fx_charge'] = 0.492/24/4
                db['P_tou_charge'] = 0.651/24/4
                db['P_tou'] = db['P'].copy()
                db['P_tou'][:] = 0.11411
                if scenarios[s]['use_pv']  == False:
                    db['R'][:] = 0
                elif house_num in [1,3,4]:
                    db['R'] = pickle.load(open('r_7', 'rb'))
                cc = 0
                for i in range(0, db['T'], 24*4):
                    cc += 1
                    if cc % 7 in [1,2,3,4,0]:
                        db['P_tou'][i+13*4:i+21*4] = 0.26771
                        db['P_ev'][i+13*4:i+21*4] = 0.16859
                        if db['E_max'] > 0 :
                            db['V'][i+h_st*4+int(m_st/15):i+h_fn*4+int(m_fn/15)] = usage/(h_fn*4+int(m_fn/15)-h_st*4-int(m_st/15))
                            db['A'][i+h_st*4+int(m_st/15):i+h_fn*4+int(m_fn/15)] = 0


                db['Pen'] = 200

                if run == False:
                    result = pickle.load(open(rf'C:\Users\sreza\Google Drive\nb\sum20\000505\results\pro_result_h_{house_num}_Scenario{s}_Rate_{p}_Sell_{sell}.pkl', 'rb'))
                else:
                    result = model_pro(db, house_num, p, sell)
                    result.to_pickle(rf'C:\Users\sreza\Google Drive\nb\sum20\000505\results\pro_result_h_{house_num}_Scenario{s}_Rate_{p}_Sell_{sell}.pkl')
                # result
            #     gfx(result, db, tbl_day = False, d1 = 180, d2 = 180, tbl_year = True, gfx_day = False, gfx_dtl = False, gfx_year = True, gfx_m = 'Non', tbl_m = 'Non')
                result.loc[:,'Scenario'] = s
                m_tab = pd.DataFrame(columns=col2, index=index, data=0.0)
                for i in np.arange(12):
                    m_tab.loc[:, 'Scenario'] = s
                    m_tab.loc[index[i],'Month']        = index[i]
                    m_tab.loc[index[i],col2]     = result.loc[t[i]:t[i+1]-1,col2].sum()
    #                 m_tab.drop(['P (Time-of-Use for GMP)', 'P (flat rate for PRO)', 'P (Time-of-Use rate for PRO)', "P (Time-of-Use for PRO's EV)", 'b^B', 'b^E'], axis=1, inplace=True)
                    m_tab.loc[index[i],'Peak load']        = result.loc[t[i],'Peak load']
                    m_tab.loc[index[i],'Agent']        = 'Pro'
                    m_tab.loc[index[i],'Rate']        = p
                    m_tab.loc[index[i],'Sell']        = sell
                    m_tab.loc[index[i],'Peak load']        = result.loc[t[i],'Peak load']
                    m_tab.loc[index[i],'Capacity cost']        = result.loc[t[i],'Capacity cost']
                    m_tab.loc[index[i],"GMP's cost"]       = m_tab.loc[index[i],'Capacity cost']

                if mark == 0:
                    results = m_tab
                    dres = result.iloc[T1:T2,:]
                    mark = 1
                else:
                    results = pd.concat([results, m_tab])
                    dres = pd.concat([dres, result.iloc[T1:T2,:]])

    results.drop(['P (GMP)', 'P (Flat)', 'P (ToU)', "P (ToU for EV)", 'b^B', 'b^E'], axis=1, inplace=True)

    results.loc[:,'Flow from Grid'] = results.loc[:,'x^GL'] + results.loc[:,'x^GB'] + results.loc[:,'x^GE'] -(results.loc[:,'x^RG'] + db['eta_d'] * (results.loc[:,'x^BG'] + results.loc[:,'x^EG']))
    results.loc[:,'Flow from ESS'] = db['eta_d'] * (results.loc[:,'x^BL'] + results.loc[:,'x^BG']) - (results.loc[:,'x^RB'] + results.loc[:,'x^GB'])
    results.loc[:,'Flow from EV'] = (results.loc[:,'x^EL'] + results.loc[:,'x^EG'] + results.loc[:,'V'])
    results.loc[:,'Flow to EV'] = (results.loc[:,'x^RE'] + results.loc[:,'x^GE'])
    dres.loc[:,'Flow from Grid'] = dres.loc[:,'x^GL'] + dres.loc[:,'x^GB'] + dres.loc[:,'x^GE'] - (dres.loc[:,'x^RG'] + db['eta_d'] * (dres.loc[:,'x^BG'] + dres.loc[:,'x^EG']))
    dres.loc[:,'Flow from ESS'] = db['eta_d'] * (dres.loc[:,'x^BL'] + dres.loc[:,'x^BG']) - (dres.loc[:,'x^RB'] + dres.loc[:,'x^GB'])
    dres.loc[:,'Flow from EV'] = db['eta_d'] * (dres.loc[:,'x^EL'] + dres.loc[:,'x^EG'] + dres.loc[:,'V']) - (dres.loc[:,'x^RE'] + dres.loc[:,'x^GE'])
    dres.loc[:,'Time (Month-Day/Hour:Minute)'] = dres.loc[:,'Month'].astype(str) + "-" + dres.loc[:,'Day'].astype(str) + "/" + dres.loc[:,'Hour'].astype(str) + ":" + dres.loc[:,'Minute'].astype(str)


    key_results = pd.DataFrame.from_dict(scenarios, orient='index')
    mark1 = 0
    mark2 = 0
    mark3 = 0
    for p in ['flat', 'Time-of-Use', 'Time-of-Use EV & flat', 'RTLMP', 'TOU&RTLMP']:
        for sell in [False, True]:
            tmp_key_results = pd.DataFrame.from_dict(scenarios, orient='index')
            tmp_key_results.loc[:, "Pro's cost"] = results[(results['Rate']==p) & (results['Sell']==sell)].groupby(['Scenario'])["Pro's cost"].sum()
            tmp_key_results.loc[:, "GMP's cost"] = results[(results['Rate']==p) & (results['Sell']==sell)].groupby(['Scenario'])["GMP's cost"].sum()
            for s in scenarios:
                if tmp_key_results.loc[s, 'E_max']>0:
                    tmp_key_results.loc[s, "Pro's cost"] -= results[(results['Rate']==p) & (results['Sell']==sell) & (results['Scenario']==s)].loc[:,"EV's cost"].sum()
                tmp_key_results.loc[s, 'Scenario'] = s
                tmp_key_results.loc[s, 'Rate'] = p
                tmp_key_results.loc[s, 'Sell'] = sell
                tmp_key_results.loc[s, 'Agent'] = 'Pro'
                tmp_key_results.loc[s, 'saving']     = tmp_key_results.loc['s0', "Pro's cost"] - tmp_key_results.loc[s, "Pro's cost"]
                returns = [-tmp_key_results.loc[s, 'price']]+N*[tmp_key_results.loc[s, 'saving']]
                tmp_key_results.loc[s, 'npv (%6.19)'] = npf.npv(ii, returns)
                tmp_key_results.loc[s, 'internal rate of return (%)'] = 100 * npf.irr(returns)    
                npv = pd.DataFrame(columns= ['Agent', 'Rate', 'Sell', 'Scenario', 'Actualization Rate (%)', 'NPV of Saving ($)'], index= np.arange(21), data=0.0)
                for i in np.arange(21):
                    npv.loc[i, 'Agent'] = 'Pro'
                    npv.loc[i, 'Rate'] = p
                    npv.loc[i, 'Sell'] = sell
                    npv.loc[i, 'Scenario'] = s
                    npv.loc[i, 'Actualization Rate (%)'] = i/2
                    npv.loc[i, 'NPV of Saving ($)'] = npf.npv(0.01*i/2, returns)
                if mark2 == 0:
                    npvs = npv
                    mark2 = 1
                else:
                    npvs = pd.concat([npvs, npv])
            if mark1 == 0:
                key_results = tmp_key_results
                mark1 = 1
            else:
                key_results = pd.concat([key_results, tmp_key_results])
                
            for s in scenarios:    
                npv2 = pd.DataFrame(columns= ['Agent', 'Rate', 'Sell', 'Scenario', 'Life time of the product(s)', 'NPV of Saving ($)'], index= np.arange(11), data=0.0)
                for i in np.arange(11):
                    tmp1 = key_results[(key_results['Rate']==p) & (key_results['Sell']==sell) & (key_results['Scenario']==s)].loc[:,"price"].sum()
                    tmp2 = key_results[(key_results['Rate']==p) & (key_results['Sell']==sell) & (key_results['Scenario']==s)].loc[:,"saving"].sum()
                    returns = [-tmp1]+(i+10)*[tmp2]
                    npv2.loc[i, 'Agent'] = 'Pro'
                    npv2.loc[i, 'Rate'] = p
                    npv2.loc[i, 'Sell'] = sell
                    npv2.loc[i, 'Scenario'] = s
                    npv2.loc[i, 'Life time of the product(s)'] = i+10
                    npv2.loc[i, 'NPV of Saving ($)'] = npf.npv(ii, returns)
                if mark3 == 0:
                    npvs2 = npv2
                    mark3 = 1
                else:
                    npvs2 = pd.concat([npvs2, npv2])
    return results, dres, key_results, npvs, npvs2



def model_gmp(db, num):
    dt = pickle.load(open(r'C:\Users\sreza\Google Drive\nb\sum20\0625\results\date.pkl', 'rb'))
    # getting parameters
    tk = [-1, 2975, 5664, 8635, 11515, 14491, 17371, 20347, 23323, 26203, 29179, 32063, 34943]
    K = 12
    E_max = db['E_max']
    E_min = db['E_min']
    B_max = db['B_max']
    B_min = db['B_min']
    T = db['T']
    U_c = db['U_c']
    U_d = db['U_d']
    eta_c = db['eta_c']
    eta_d = db['eta_d']
    R = db['R']
    L = db['L']
    A = db['A']
    V = db['V']
    Pen = db['Pen']
    # Model formulation
    m = gp.Model('gmp')
    # Decision Vars
    x_GL   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^GL')
    x_GB   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^GB')
    x_GE   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^GE')
    x_RL   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^RL')
    x_RB   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^RB')
    x_RE   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^RE')
    x_RG   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^RG')
    x_BL   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^BL')
    x_EL   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^EL')
    x_BG   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^BG')
    x_EG   = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^EG')
    x_PEAK = m.addVars(K, vtype=GRB.CONTINUOUS, name='x^PEAK')
    x_loss = m.addVars(T, vtype=GRB.CONTINUOUS, name='x^loss')
    b_E    = m.addVars(T, vtype=GRB.CONTINUOUS, name='b^E')
    b_B    = m.addVars(T, vtype=GRB.CONTINUOUS, name='b^B')
    # Constraints
    m.addConstrs((x_GL[t] + eta_d * (x_BL[t] + x_EL[t]) + x_RL[t] + x_loss[t] == L[t] for t in range(T)), name='c_load_2')
    m.addConstrs((x_BL[t] + x_BG[t] <= b_B[t] for t in range(T)), name='c_bat_state_2')
    m.addConstrs((x_EL[t] + x_EG[t] <= b_E[t] * A[t] for t in range(T)), name='c_ev_state_2')
    m.addConstrs((x_RE[t] + x_GE[t] <= E_max * A[t] for t in range(T)), name='c_ev_2_state_2')
    m.addConstrs((x_BL[t] + x_BG[t] + x_EL[t] + x_EG[t] <= U_d for t in range(T)), name='c_dch_max_2')
    m.addConstrs((x_GB[t] + x_RB[t] + x_GE[t] + x_RE[t] <= U_c for t in range(T)), name='c_ch_max_2')
    m.addConstrs((b_B[t] <= B_max for t in range(T)), name='c_b_B_max_2')
    m.addConstrs((b_B[t] >= B_min for t in range(T)), name='c_b_B_min_2')
    m.addConstrs((b_E[t] <= E_max for t in range(T)), name='c_b_E_max_2')
    m.addConstrs((b_E[t] >= E_min for t in range(T)), name='c_b_E_min_2')
    m.addConstrs((x_RL[t] + x_RB[t] + x_RE[t] + x_RG[t] <= R[t] for t in range(T)), name='c_gen_cap_2')
    for k in range(K):
        for t in range(tk[k]+1,tk[k+1]+1):
            m.addConstr((x_PEAK[k] >= x_GB[t] +  x_GE[t] + x_GL[t]), name='maxconstr')
    m.addConstrs((b_B[t] + eta_c * (x_GB[t] + x_RB[t]) - (x_BL[t] + x_BG[t]) == b_B[t+1] for t in range(T-1)), name='c_next_state_b_B_2')
    m.addConstrs((b_E[t] + eta_c * (x_GE[t] + x_RE[t]) - (x_EL[t] + x_EG[t]) - (V[t] / eta_d) == b_E[t+1] for t in range(T-1)), name='c_next_state_b_E_2')
    # Objective
    m.setObjective(gp.quicksum(db['P_gmp'][t] * (1.0001 * (x_GB[t] + x_GE[t] + x_GL[t])- 0.9999 * (x_RG[t] + eta_d * (x_BG[t]+ x_EG[t]))) 
                               + Pen * x_loss[t] for t in range(T)) +gp.quicksum((1.35*5+9)* 4 * x_PEAK[k] for k in range(K)), GRB.MINIMIZE)
    # Optimize
    m.optimize()
    # result
#     m.write("gmp-output_%g.sol" % num)
    columns = "x^GL, x^GB, x^GE, x^RL, x^RB, x^RE, x^RG, x^BL, x^EL, x^BL * eta_d, x^EL * eta_d, x^BG, x^EG, x^BG * eta_d, x^EG * eta_d, b^B, b^E, x^loss, Peak load, V, R, L, Capacity cost, Cost of load covering by GMP, GMP's cost, Pro's cost (buy : flat & sell : flat), Pro's cost (buy : flat & sell : Time-of-Use), Pro's cost (buy : Time-of-Use & sell : flat), Pro's cost (buy : Time-of-Use & sell : Time-of-Use), Pro's cost (buy : Time-of-Use EV & sell : flat), Pro's cost (buy : Time-of-Use EV & sell : Time-of-Use) , Loss cost, EV's cost, Agent, Rate".split(', ')
    result = pd.DataFrame(columns=columns, index=range(T), data=0.0)
    GL = np.zeros(T)
    GB = np.zeros(T)
    GE = np.zeros(T)
    RL = np.zeros(T)
    RB = np.zeros(T)
    RE = np.zeros(T)
    RG = np.zeros(T)
    BL = np.zeros(T)
    EL = np.zeros(T)
    BG = np.zeros(T)
    EG = np.zeros(T)
    ls = np.zeros(T)
    xb_B = np.zeros(T)
    xb_E = np.zeros(T)
    peak = np.zeros(T)
    l_cost = np.zeros(T)
    ev_cost = np.zeros(T)
    fx_fx_cost = np.zeros(T)
    tou_fx_cost = np.zeros(T)
    evfx_fx_cost = np.zeros(T)
    fx_tou_cost = np.zeros(T)
    tou_tou_cost = np.zeros(T)
    evfx_tou_cost = np.zeros(T)
    cap_cost = np.zeros(T)
    gmp_cost = np.zeros(T)
#     ttl_cost = np.zeros(T)
    ls_cost = np.zeros(T)

    for k in range(K):
        peak[tk[k]+1:tk[k+1]+1] = 4 * x_PEAK[k].x
        cap_cost[tk[k]+1:tk[k+1]+1] = (1.35*5+9) * 4 * x_PEAK[k].x
    
    for t in range(T):
        GL[t] = x_GL[t].x
        GB[t] = x_GB[t].x
        GE[t] = x_GE[t].x
        RL[t] = x_RL[t].x
        RB[t] = x_RB[t].x
        RE[t] = x_RE[t].x
        RG[t] = x_RG[t].x
        BL[t] = x_BL[t].x
        EL[t] = x_EL[t].x
        BG[t] = x_BG[t].x
        EG[t] = x_EG[t].x
        ls[t] = x_loss[t].x
        xb_B[t] = b_B[t].x
        xb_E[t] = b_E[t].x
        l_cost[t] = L[t] * db['P_gmp'][t]
        fx_fx_cost[t] = db['P_fx_charge'] + (GB[t] + GE[t] + GL[t] - RG[t] - db['eta_d'] * (BG[t] + EG[t])) * db['P_fx'][t]
        fx_tou_cost[t] = db['P_fx_charge'] + (GB[t] + GE[t] + GL[t])  * db['P_fx'][t] - (RG[t] + db['eta_d'] * (BG[t] + EG[t])) * db['P_tou'][t]
        tou_fx_cost[t] = db['P_tou_charge'] + (GB[t] + GE[t] + GL[t]) * db['P_tou'][t]-(RG[t] + db['eta_d'] * (BG[t] + EG[t])) * db['P_fx'][t]
        tou_tou_cost[t] = db['P_tou_charge'] + (GB[t] + GE[t] + GL[t]) * db['P_tou'][t]-(RG[t] + db['eta_d'] * (BG[t] + EG[t])) * db['P_tou'][t]
        evfx_fx_cost[t] = db['P_fx_charge'] + (GB[t]  + GL[t] - RG[t] - db['eta_d']*(BG[t] + EG[t]))*db['P_fx'][t] + GE[t] * db['P_ev'][t]
        evfx_tou_cost[t] = db['P_fx_charge'] + (GB[t]  + GL[t]) *db['P_fx'][t] - (RG[t] + db['eta_d']*(BG[t] + EG[t])) * db['P_tou'][t] + GE[t] * db['P_ev'][t]
        gmp_cost[t] = db['P_gmp'][t] * (GB[t] + GE[t] + GL[t] - RG[t] - db['eta_d'] * (BG[t] + EG[t]))
#         ttl[t] = 0
        ev_cost[t] = db['P_gmp'][t] * V[t] / eta_c/ eta_d
        ls_cost[t] = Pen * ls[t]
     

    result.loc[:,'Agent'] = 'GMP'
    result.loc[:,'Rate'] = 'Time-of-Use for GMP'
    result.loc[:,'x^GL'] = GL
    result.loc[:,'x^GB'] = GB
    result.loc[:,'x^GE'] = GE
    result.loc[:,'x^RL'] = RL
    result.loc[:,'x^RB'] = RB
    result.loc[:,'x^RE'] = RE
    result.loc[:,'x^RG'] = RG
    result.loc[:,'x^BL'] = BL
    result.loc[:,'x^EL'] = EL
    result.loc[:,'x^BG'] = BG
    result.loc[:,'x^EG'] = EG
    result.loc[:,'x^BL * eta_d'] = BL * db['eta_d']
    result.loc[:,'x^BG * eta_d'] = BG * db['eta_d']
    result.loc[:,'x^EL * eta_d'] = EL * db['eta_d']
    result.loc[:,'x^EG * eta_d'] = EG * db['eta_d']
    result.loc[:,'x^loss'] = ls
    result.loc[:,'Peak load'] = peak
    result.loc[:,'V'] = V
    result.loc[:,'P_GMP'] = db['P_gmp']
    result.loc[:,'b^B'] = xb_B
    result.loc[:,'b^E'] = xb_E
    result.loc[:,'R'] = R
    result.loc[:,'L'] = L
    result.loc[:,'Default cost of load covering'] = l_cost
    result.loc[:,"Pro's cost (buy : flat & sell : flat)"] = fx_fx_cost
    result.loc[:,"Pro's cost (buy : flat & sell : Time-of-Use)"] = fx_tou_cost
    result.loc[:,"Pro's cost (buy : Time-of-Use & sell : flat)"] = tou_fx_cost
    result.loc[:,"Pro's cost (buy : Time-of-Use & sell : Time-of-Use)"] = tou_tou_cost
    result.loc[:,"Pro's cost (buy : Time-of-Use EV & sell : flat)"] = evfx_fx_cost
    result.loc[:,"Pro's cost (buy : Time-of-Use EV & sell : Time-of-Use)"] = evfx_tou_cost
    result.loc[:,'Capacity cost'] = cap_cost
    result.loc[:,'Cost of load covering by GMP'] = gmp_cost
#     result.loc[:,"GMP's cost"] = ttl
    result.loc[:,"EV's cost"] = ev_cost
    result.loc[:,'Loss cost'] = ls_cost
           
    result = pd.concat([dt, result], axis = 1)

    
    return result

def gmp_ctrl(run,house_num):

    t = [0, 2977, 5665, 8637, 11517, 14493, 17373, 20349, 23325, 26205, 29181, 32065, 34945]
    index = 'Jan.Feb.Mar.Apr.May.Jun.Jul.Aug.Sep.Oct.Nov.Dec'.split('.')
    col2 = "x^GL, x^GB, x^GE, x^RL, x^RB, x^RE, x^RG, x^BL, x^EL, x^BL * eta_d, x^EL * eta_d, x^BG, x^EG, x^BG * eta_d, x^EG * eta_d, b^B, b^E, x^loss, Peak load, V, R, L, Default cost of load covering, Pro's cost (buy : flat & sell : flat), Pro's cost (buy : flat & sell : Time-of-Use), Pro's cost (buy : Time-of-Use & sell : flat), Pro's cost (buy : Time-of-Use & sell : Time-of-Use), Pro's cost (buy : Time-of-Use EV & sell : flat), Pro's cost (buy : Time-of-Use EV & sell : Time-of-Use), Capacity cost, Cost of load covering by GMP, GMP's cost, Loss cost, EV's cost, Agent, Rate".split(', ')

    N= 15
    ii = 0.0619


    d1 = 61
    d2 = d1+6
    T1 = 24*4*(d1-1)
    T2 = 24*4*d2

    scenarios = {
      "s0" : {'B_max': 0,    'B_min': 0,  'U_c': 0,    'U_d': 0,    'E_max': 0,  'E_min': 0,  'use_pv': False, 'price': 0},
      "s1" : {'B_max': 40.5, 'B_min': 0,  'U_c': 3.75, 'U_d': 3.75, 'E_max': 0,  'E_min': 0,  'use_pv': False, 'price': 27300},
      "s2" : {'B_max': 27,   'B_min': 0,  'U_c': 2.5,  'U_d': 2.5,  'E_max': 0,  'E_min': 0,  'use_pv': True,  'price': 35125},
      "s3" : {'B_max': 0,    'B_min': 0,  'U_c': 1.9,  'U_d': 1.9,  'E_max': 60, 'E_min': 24, 'use_pv': False, 'price': 4000},
      "s4" : {'B_max': 0,    'B_min': 0,  'U_c': 1.9,  'U_d': 1.9,  'E_max': 60, 'E_min': 24, 'use_pv': True,  'price': 17145},
      "s5" : {'B_max': 10,   'B_min': 0,  'U_c': 1.9,  'U_d': 1.9,  'E_max': 60, 'E_min': 24, 'use_pv': True,  'price': 23724},
    }
    
    for s in scenarios:
    #     print(f'''House number is: {house_num}''')
#         print(scenarios[s])
     
        use_pv = scenarios[s]['use_pv']
        bat = 'Always'
        db = pickle.load(open(rf'C:\Users\sreza\Google Drive\nb\sum20\0625\results\db_h_{house_num}_RES_{use_pv}_ESS_{bat}.pkl', 'rb'))
        h_st = 8
        m_st = 45
        h_fn = 17
        m_fn = 15
        usage = 10
        for i in scenarios[s]:
            db[i] = scenarios[s][i]

        db['P_ev'] = db['P'].copy()
        db['P_ev'][:] = 0.12831
        db['P_gmp'] = db['P'].copy()
        db['P_fx'] = db['P'].copy()
        db['P_fx'][:] = 0.16859
        db['P_fx_charge'] = 0.492/24/4
        db['P_tou_charge'] = 0.651/24/4
        db['P_tou'] = db['P'].copy()
        db['P_tou'][:] = 0.11411
        if scenarios[s]['use_pv']  == False:
            db['R'][:] = 0
        elif house_num in [1,3,4]:
            db['R'] = pickle.load(open('r_7', 'rb'))
        cc = 0
        for i in range(0, db['T'], 24*4):
            cc += 1
            if cc % 7 in [1,2,3,4,5]:
                db['P_tou'][i+13*4:i+21*4] = 0.26771
                db['P_ev'][i+13*4:i+21*4] = 0.16859
                if db['E_max'] > 0 :
                    db['V'][i+h_st*4+int(m_st/15):i+h_fn*4+int(m_fn/15)] = usage/(h_fn*4+int(m_fn/15)-h_st*4-int(m_st/15))
                    db['A'][i+h_st*4+int(m_st/15):i+h_fn*4+int(m_fn/15)] = 0


        db['Pen'] = 200
        if run == False:
            result = pickle.load(open(rf'C:\Users\sreza\Google Drive\nb\sum20\1230\results\gmp_result_h_{house_num}_Scenario_{s}.pkl', 'rb'))
        else:
            result = model_gmp(db, house_num)
            result.to_pickle(rf'C:\Users\sreza\Google Drive\nb\sum20\1230\results\gmp_result_h_{house_num}_Scenario_{s}.pkl')
        # result
        result.loc[:,'Scenario'] = s
        m_tab = pd.DataFrame(columns=col2, index=index, data=0.0)
        for i in np.arange(12):
            m_tab.loc[:, 'Scenario'] = s
            m_tab.loc[index[i],'Month']        = index[i]
            m_tab.loc[index[i],col2]     = result.loc[t[i]:t[i+1]-1,col2].sum()
            m_tab.loc[index[i],'Agent']        = 'GMP'
            m_tab.loc[index[i],'Rate']        = 'Time-of-Use for GMP'
            m_tab.loc[index[i],'Trade']        = True
            m_tab.loc[index[i],'Peak load']        = result.loc[t[i],'Peak load']
            m_tab.loc[index[i],'Capacity cost']        = result.loc[t[i],'Capacity cost']
            m_tab.loc[index[i],"GMP's cost"]       = result.loc[t[i],'Capacity cost'] + result.loc[t[i]:t[i+1]-1,'Cost of load covering by GMP'].sum()
        if s == 's0':
            results = m_tab
            dres = result.iloc[T1:T2,:]
        else:
            results = pd.concat([results, m_tab])
            dres = pd.concat([dres, result.iloc[T1:T2,:]])
    results.drop(['b^B', 'b^E'], axis=1)
    results.loc[:,'Flow from Grid'] = results.loc[:,'x^GL'] + results.loc[:,'x^GB'] + results.loc[:,'x^GE'] -(results.loc[:,'x^RG'] + db['eta_d'] * (results.loc[:,'x^BG'] + results.loc[:,'x^EG']))
    results.loc[:,'Flow from ESS'] = db['eta_d'] * (results.loc[:,'x^BL'] + results.loc[:,'x^BG']) - (results.loc[:,'x^RB'] + results.loc[:,'x^GB'])
    results.loc[:,'Flow from EV'] = (results.loc[:,'x^EL'] + results.loc[:,'x^EG'] + results.loc[:,'V'])
    results.loc[:,'Flow to EV'] = (results.loc[:,'x^RE'] + results.loc[:,'x^GE'])
    # dres.loc[:,'Total cost'] = dres.loc[:,'V2G cost'] + dres.loc[:,'Peak cost']
    dres.loc[:,'Flow from Grid'] = dres.loc[:,'x^GL'] + dres.loc[:,'x^GB'] + dres.loc[:,'x^GE'] -(dres.loc[:,'x^RG'] + db['eta_d'] * (dres.loc[:,'x^BG'] + dres.loc[:,'x^EG']))
    dres.loc[:,'Flow from ESS'] = db['eta_d'] * (dres.loc[:,'x^BL'] + dres.loc[:,'x^BG']) - (dres.loc[:,'x^RB'] + dres.loc[:,'x^GB'])
    dres.loc[:,'Flow from EV'] = db['eta_d'] * (dres.loc[:,'x^EL'] + dres.loc[:,'x^EG'] + dres.loc[:,'V']) - (dres.loc[:,'x^RE'] + dres.loc[:,'x^GE'])
    dres.loc[:,'Time (Month-Day/Hour:Minute)'] = dres.loc[:,'Month'].astype(str) + "-" + dres.loc[:,'Day'].astype(str) + "/" + dres.loc[:,'Hour'].astype(str) + ":" + dres.loc[:,'Minute'].astype(str)
    key_results = pd.DataFrame.from_dict(scenarios, orient='index')
    key_results.loc[:, "GMP's cost"] = results.groupby(['Scenario'])["GMP's cost"].sum()
    key_results.loc[:, "Pro's cost (buy : flat & sell : flat)"] = results.groupby(['Scenario'])["Pro's cost (buy : flat & sell : flat)"].sum()
    key_results.loc[:, "Pro's cost (buy : flat & sell : Time-of-Use)"] = results.groupby(['Scenario'])["Pro's cost (buy : flat & sell : Time-of-Use)"].sum()
    key_results.loc[:, "Pro's cost (buy : Time-of-Use & sell : flat)"] = results.groupby(['Scenario'])["Pro's cost (buy : Time-of-Use & sell : flat)"].sum()
    key_results.loc[:, "Pro's cost (buy : Time-of-Use & sell : Time-of-Use)"] = results.groupby(['Scenario'])["Pro's cost (buy : Time-of-Use & sell : Time-of-Use)"].sum()
    key_results.loc[:, "Pro's cost (buy : Time-of-Use EV & sell : flat)"] = results.groupby(['Scenario'])["Pro's cost (buy : Time-of-Use EV & sell : flat)"].sum()
    key_results.loc[:, "Pro's cost (buy : Time-of-Use EV & sell : Time-of-Use)"] = results.groupby(['Scenario'])["Pro's cost (buy : Time-of-Use EV & sell : Time-of-Use)"].sum()
    key_results.loc[:, "System cost"] = results.groupby(['Scenario'])["Capacity cost"].sum()
    for s in scenarios:
        key_results.loc[s, 'Scenario'] = s
        key_results.loc[s, 'Rate'] = 'Time-of-Use for GMP'
        key_results.loc[s, 'Trade'] = True
        key_results.loc[s, 'Agent'] = 'GMP'
        if key_results.loc[s, 'E_max']>0:
            key_results.loc[s, "GMP's cost"] -= results[results['Scenario']==s].loc[:,"EV's cost"].sum()
            results.groupby(['Scenario'])["EV's cost"].sum()
        key_results.loc[s, 'saving']     = key_results.loc['s0', "GMP's cost"] - key_results.loc[s, "GMP's cost"]
        returns = [-key_results.loc[s, 'price']]+N*[key_results.loc[s, 'saving']]
        key_results.loc[s, 'npv (%6.19)'] = npf.npv(ii, returns)
        key_results.loc[s, 'internal rate of return (%)'] = 100 * npf.irr(returns)
        returns = [-key_results.loc[s, 'price']]+N*[key_results.loc[s, 'saving']]
        npv = pd.DataFrame(columns= ['Scenario', 'Actualization Rate (%)', 'NPV of Saving ($)'], index= np.arange(21), data=0.0)
        for i in np.arange(21):
            npv.loc[i, 'Scenario'] = s
            npv.loc[i, 'Actualization Rate (%)'] = i/2
            npv.loc[i, 'NPV of Saving ($)'] = npf.npv(0.01*i/2, returns)
        npv2 = pd.DataFrame(columns= ['Scenario', 'Life time of the product(s)', 'NPV of Saving ($)'], index= np.arange(11), data=0.0)
        for i in np.arange(11):
            returns = [-key_results.loc[s, 'price']]+(i+10)*[key_results.loc[s, 'saving']]
            npv2.loc[i, 'Scenario'] = s
            npv2.loc[i, 'Life time of the product(s)'] = i+10
            npv2.loc[i, 'NPV of Saving ($)'] = npf.npv(ii, returns)
        if s == 's0':
            npvs = npv
            npvs2 = npv2
        else:
            npvs = pd.concat([npvs, npv])
            npvs2 = pd.concat([npvs2, npv2])
    return results, dres, key_results, npvs, npvs2

