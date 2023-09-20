import network_diffusion as nd
import numpy as np


# ALFA, BETA are SIR parameters with values from sec. 2.3.1 in
# https://www.mdpi.com/1099-4300/25/2/231.
# GAMMA and EPSILON stands from awareness induced by mass media.
# DELTA is a mass media factor + rate of symptomatic infections from paper: 
# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0249090.
ALPHA = 0.19
BETA = 0.10
GAMMA = 0.05
DELTA = GAMMA + 1 - 0.3
EPSILON = GAMMA


def get_covid_model(l: int):

    # define processes, allowed states and initial % of actors in that states
    phenomena = {"contagion": [["S", "I", "R"], [95, 2.5, 2.5]], "awareness": [["U", "A"], [95, 5]]}

    # wrap them into a compartments 
    comp_g = nd.models.CompartmentalGraph()
    for phenomenon, [states, budget] in phenomena.items():
        comp_g.add(process_name=phenomenon, states=states)  # name of process
        comp_g.seeding_budget.update({phenomenon: budget})  # initial %s
    comp_g.compile(background_weight=0)

    # set up weights of transitions for SIR and unaware
    comp_g.set_transition_fast("contagion.S", "contagion.I", ("awareness.U", ), ALPHA)
    comp_g.set_transition_fast("contagion.I", "contagion.R", ("awareness.U", ), BETA)

    # set up weights of transitions for SIR and aware
    comp_g.set_transition_fast("contagion.S", "contagion.I", ("awareness.A", ), ALPHA * np.exp(-1 * l))  # a * {1, 0.35, 0.1} nic, maseczka lub sucial distancing, pełne obostrzenia
    comp_g.set_transition_fast("contagion.I", "contagion.R", ("awareness.A", ), BETA * np.exp(l))

    # set up weights of transitions for UA and suspected
    comp_g.set_transition_fast("awareness.U", "awareness.A", ("contagion.S", ), GAMMA)

    # set up weights of transitions for UA and infected
    comp_g.set_transition_fast("awareness.U", "awareness.A", ("contagion.I", ), DELTA)

    # set up weights of transitions for UA and removed
    comp_g.set_transition_fast("awareness.U", "awareness.A", ("contagion.R", ), EPSILON)

    # create a DSAA model that implements propagation mechanism we are going to use
    model = nd.models.DSAAModel(compartmental_graph=comp_g)
    
    return model
