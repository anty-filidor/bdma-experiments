import network_diffusion as nd
import numpy as np

from network_diffusion.models.utils.compartmental import CompartmentalGraph


ALPHA = 0.3
BETA = 0.1
GAMMA = 0.25
DELTA = 0.5
EPSILON = 1


def get_flu_model(l: int):

    # define processes, allowed states and initial % of actors in that states
    phenomena = {"contagion": [["S", "I", "R"], [90, 5, 5]], "awareness": [["U", "A"], [95, 5]]}

    # wrap them into a compartments 
    comp_g = CompartmentalGraph()
    for phenomenon, [states, budget] in phenomena.items():
        comp_g.add(process_name=phenomenon, states=states)  # name of process
        comp_g.seeding_budget.update({phenomenon: budget})  # initial %s
    comp_g.compile(background_weight=0)

    # set up weights of transitions for SIR and unaware
    comp_g.set_transition_fast("contagion.S", "contagion.I", ("awareness.U", ), ALPHA)
    comp_g.set_transition_fast("contagion.I", "contagion.R", ("awareness.U", ), BETA)

    # set up weights of transitions for SIR and aware
    comp_g.set_transition_fast("contagion.S", "contagion.I", ("awareness.A", ), ALPHA * np.exp(-1 * l))
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
