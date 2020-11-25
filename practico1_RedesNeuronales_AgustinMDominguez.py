# Agustin Marcelo Dominguez - Nov 2020

def line(ch = '-', msg=''):
    for _ in range(80):
        print(ch, end='')
    print('\n\t' + msg)

line(msg="loading libraries...")
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_context('talk')
line(msg="Starting...")

# MODEL PARAMETERS
ALPHA = 0.1
BETA =  0.02
GAMMA = 0.3
DELTA = 0.01

# INTEGRATION PARAMETERS
ITGR_STEP = 0.01
ITGR_INIT = 0
ITGR_STOP = 300

# SAVE IMAGE
SAVE = True

# COLORS
colors = ['#000000', '#1f77b4', "#b7c400", "#5c1254", "#330003", "#363636", "#178c0d", "#08d4bf", "#ff5938", "#ff8c00" ]
itgr_span = (ITGR_INIT, ITGR_STOP)
itgr_eval = np.arange(ITGR_INIT, ITGR_STOP, ITGR_STEP)

# INITIAL CONDITIONS
initial_conditions = [
    (40, 9 ),
    (70, 70),
    (10, 10),
    (1 , 1 ),
    (50, 10),
    (27, 5 ),
    (25, 5 ),
    (22, 5 )
]

def lotkaVolterra(t, y):
    rabbits, wolves = y
    drabbits_dtime = ALPHA*rabbits - BETA*rabbits*wolves
    dwolves_dtime = -GAMMA*wolves + DELTA*rabbits*wolves
    dy_dt = (drabbits_dtime, dwolves_dtime)
    return dy_dt

global_solutions = []
for y0 in initial_conditions:
    line(msg=f"Finding solution for initial condition: = {y0}")
    solution = solve_ivp(lotkaVolterra, t_span=itgr_span, y0=y0, t_eval=itgr_eval)
    global_solutions.append(solution)
print("All solutions found.")

def plotPopulationComparison(rabbitPopulation, wolvesPopulation, logscale=False, save=False):
    y0 = (rabbitPopulation, wolvesPopulation)
    solution = solve_ivp(lotkaVolterra, t_span=itgr_span, y0=y0, t_eval=itgr_eval)
    print(solution.message, " Initial condition:", y0)
    _, ax = plt.subplots(figsize=(6, 4))
    ax.set_title('Initial Condition: ({:.1f}, {:.1f})'.format(rabbitPopulation, wolvesPopulation))
    ax.set_ylabel('Population')
    if (logscale):
        ax.set_yscale("log")
    ax.set_xlabel('Time')   
    ax.plot(solution.t, solution.y[0], label='Rabbits')
    ax.plot(solution.t, solution.y[1], label='Wolves')
    ax.legend()
    plt.tight_layout()
    if (save):
        logstr = "log" if logscale else "lin"
        plt.savefig(f"lotkaVolterra_{logstr}_{rabbitPopulation}_{wolvesPopulation}.png", dpi=300)
        print("Imaged Saved")
    else:
        plt.show()

def plotComparisonAllInitialConditions(save=SAVE):
    for condition in initial_conditions:
        plotPopulationComparison(condition[0], condition[1], logscale=False, save=SAVE)
        plotPopulationComparison(condition[0], condition[1], logscale=True, save=SAVE)

def drawGraph1(save=False):
    line(msg="Drawing Graph 1")
    r_values = np.linspace(0, 60, 30)
    s_values = np.linspace(0, 60, 30)
    R, S = np.meshgrid(r_values, s_values)
    U, V = lotkaVolterra(None, [R, S])
    _, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-2., 60.)
    ax.set_ylim(-2., 60.)
    ax.set_ylabel('Wolves')
    ax.set_xlabel('Rabbits')
    ax.quiver(R, S, U, V)
    for i in range(len(initial_conditions)):
        sol = global_solutions[i]
        ax.plot(sol.y[0], sol.y[1], color=colors[i])
    ax.plot(30, 5, '.', color="#000000")
    ax.plot(0, 0, 'o', color="#000000")
    if save:
        plt.savefig('fase1.png', dpi=300)
    else:
        plt.show()

def drawGraph2(save=False):
    line(msg="Drawing Graph 2")
    r_values = np.linspace(0, 60, 30)
    s_values = np.linspace(0, 10, 30)
    R, S = np.meshgrid(r_values, s_values)
    U, V = lotkaVolterra(None, [R, S])
    _, ax = plt.subplots(figsize=(20, 10))
    ax.set_xlim(-2., 60.)
    ax.set_ylim(-0.5, 10.)
    ax.set_ylabel('Wolves')
    ax.set_xlabel('Rabbits')
    ax.quiver(R, S, U, V)
    for i in range(len(initial_conditions)):
        sol = global_solutions[i]
        ax.plot(sol.y[0], sol.y[1], color=colors[i])
    
    ax.plot(30, 5, '.', color="#000000")
    ax.plot(0, 0, 'o', color="#000000")
    if save:
        plt.savefig('fase2.png', dpi=300)
    else:
        plt.show()

drawGraph1(save=True)
drawGraph2(save=True)
plotComparisonAllInitialConditions(save=True)

line(msg="Completed")