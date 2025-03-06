import numpy as np
from an_cockrell import AnCockrellModel, EndoType, EpiType

from util import compute_desired_epi_counts, smooth_random_field

################################################################################


def modify_model(
    model: AnCockrellModel,
    desired_state: np.ndarray,
    *,
    ignore_state_vars: bool = False,
    variational_params,
    state_vars,
    state_var_indices,
    verbose: bool = False,
):
    """
    Modify a model's microstate to fit a given macrostate

    :param model: model instance (encodes microstate)
    :param desired_state: desired macrostate for the model
    :param ignore_state_vars: if True, only alter parameters, not state variables
    :param verbose: if true, print diagnostic messages
    :param state_var_indices:
    :param state_vars:
    :param variational_params:
    :return: None
    """
    np.abs(desired_state, out=desired_state)  # in-place absolute value

    for param_idx, param_name in enumerate(variational_params, start=len(state_vars)):
        prev_value = getattr(model, param_name, None)
        assert prev_value is not None
        if isinstance(prev_value, int):
            setattr(model, param_name, round(float(desired_state[param_idx])))
        else:
            setattr(model, param_name, desired_state[param_idx])

    if ignore_state_vars:
        return

    # it's worth pointing out that, because of the cleanup-below-a-threshold code,
    # these reset-by-scaling fields may not go to quite the right thing on the next
    # time step. I'm not seeing an easy fix/alternative here, so ¯\_(ツ)_/¯ ?

    # another problem is what to do when you have zero totals, punting here as well

    desired_t1ifn = desired_state[state_var_indices["total_T1IFN"]]
    if model.total_T1IFN > 0:
        model.T1IFN *= desired_t1ifn / model.total_T1IFN
    else:
        random_field = smooth_random_field(model.geometry)
        model.T1IFN[:] = random_field * desired_t1ifn / np.sum(random_field)

    desired_tnf = desired_state[state_var_indices["total_TNF"]]
    if model.total_TNF > 0:
        model.TNF *= desired_tnf / model.total_TNF
    else:
        random_field = smooth_random_field(model.geometry)
        model.TNF[:] = random_field * desired_tnf / np.sum(random_field)

    desired_ifn_g = desired_state[state_var_indices["total_IFNg"]]
    if model.total_IFNg > 0:
        model.IFNg *= desired_ifn_g / model.total_IFNg
    else:
        random_field = smooth_random_field(model.geometry)
        model.IFNg[:] = random_field * desired_ifn_g / np.sum(random_field)

    desired_il1 = desired_state[state_var_indices["total_IL1"]]
    if model.total_IL1 > 0:
        model.IL1 *= desired_il1 / model.total_IL1
    else:
        random_field = smooth_random_field(model.geometry)
        model.IL1[:] = random_field * desired_il1 / np.sum(random_field)

    desired_il6 = desired_state[state_var_indices["total_IL6"]]
    if model.total_IL6 > 0:
        model.IL6 *= desired_il6 / model.total_IL6
    else:
        random_field = smooth_random_field(model.geometry)
        model.IL6[:] = random_field * desired_il6 / np.sum(random_field)

    desired_il8 = desired_state[state_var_indices["total_IL8"]]
    if model.total_IL8 > 0:
        model.IL8 *= desired_il8 / model.total_IL8
    else:
        random_field = smooth_random_field(model.geometry)
        model.IL8[:] = random_field * desired_il8 / np.sum(random_field)

    desired_il10 = desired_state[state_var_indices["total_IL10"]]
    if model.total_IL10 > 0:
        model.IL10 *= desired_il10 / model.total_IL10
    else:
        random_field = smooth_random_field(model.geometry)
        model.IL10[:] = random_field * desired_il10 / np.sum(random_field)

    desired_il12 = desired_state[state_var_indices["total_IL12"]]
    if model.total_IL12 > 0:
        model.IL12 *= desired_il12 / model.total_IL12
    else:
        random_field = smooth_random_field(model.geometry)
        model.IL12[:] = random_field * desired_il12 / np.sum(random_field)

    desired_il18 = desired_state[state_var_indices["total_IL18"]]
    if model.total_IL18 > 0:
        model.IL18 *= desired_il18 / model.total_IL18
    else:
        random_field = smooth_random_field(model.geometry)
        model.IL18[:] = random_field * desired_il18 / np.sum(random_field)

    desired_total_extracellular_virus = desired_state[
        state_var_indices["total_extracellular_virus"]
    ]
    if model.total_extracellular_virus > 0:
        model.extracellular_virus *= (
            desired_total_extracellular_virus / model.total_extracellular_virus
        )
    else:
        model.infect(int(np.rint(desired_total_extracellular_virus)))

    desired_total_intracellular_virus = desired_state[
        state_var_indices["total_intracellular_virus"]
    ]
    # TODO: how close does this get? do we have to deal with discretization error?
    if model.total_intracellular_virus > 0:
        model.epi_intracellular_virus[:] = np.rint(
            model.epi_intracellular_virus[:]
            * (desired_total_intracellular_virus / model.total_intracellular_virus),
        ).astype(int)
    # if the update clears out all virus in an infected cell, it should be healthy
    model.epithelium[
        np.where(
            (model.epithelium == EpiType.Infected)
            & (model.epi_intracellular_virus <= 0)
        )
    ] = EpiType.Healthy

    model.apoptosis_eaten_counter = int(
        np.rint(desired_state[state_var_indices["apoptosis_eaten_counter"]])
    )  # no internal state here

    ################################################################################
    # randomized change of epithelial states to match sampled distribution

    (
        desired_empty_epithelium,
        desired_healthy_epithelium,
        desired_infected_epithelium,
        desired_dead_epithelium,
        desired_apoptosed_epithelium,
    ) = compute_desired_epi_counts(desired_state, model, state_var_indices)

    # first, kill off (empty) any excess in the epithelial categories
    if model.healthy_epithelium_count > desired_healthy_epithelium:
        healthy_locations = np.where(model.epithelium == EpiType.Healthy)
        locs_to_empty = np.random.choice(
            len(healthy_locations[0]),
            model.healthy_epithelium_count - desired_healthy_epithelium,
            replace=False,
        )
        model.epithelium[
            healthy_locations[0][locs_to_empty], healthy_locations[1][locs_to_empty]
        ] = EpiType.Empty
        model.epi_intracellular_virus[
            healthy_locations[0][locs_to_empty], healthy_locations[1][locs_to_empty]
        ] = 0

    if model.infected_epithelium_count > desired_infected_epithelium:
        infected_locations = np.where(model.epithelium == EpiType.Infected)
        locs_to_empty = np.random.choice(
            len(infected_locations[0]),
            model.infected_epithelium_count - desired_infected_epithelium,
            replace=False,
        )
        model.epithelium[
            infected_locations[0][locs_to_empty], infected_locations[1][locs_to_empty]
        ] = EpiType.Empty
        model.epi_intracellular_virus[
            infected_locations[0][locs_to_empty], infected_locations[1][locs_to_empty]
        ] = 0
        # TODO: move the recalc of epi_intracellular_virus after this?

    if model.dead_epithelium_count > desired_dead_epithelium:
        dead_locations = np.where(model.epithelium == EpiType.Dead)
        locs_to_empty = np.random.choice(
            len(dead_locations[0]),
            model.dead_epithelium_count - desired_dead_epithelium,
            replace=False,
        )
        model.epithelium[
            dead_locations[0][locs_to_empty], dead_locations[1][locs_to_empty]
        ] = EpiType.Empty
        model.epi_intracellular_virus[
            dead_locations[0][locs_to_empty], dead_locations[1][locs_to_empty]
        ] = 0

    if model.apoptosed_epithelium_count > desired_apoptosed_epithelium:
        apoptosed_locations = np.where(model.epithelium == EpiType.Apoptosed)
        locs_to_empty = np.random.choice(
            len(apoptosed_locations[0]),
            model.apoptosed_epithelium_count - desired_apoptosed_epithelium,
            replace=False,
        )
        model.epithelium[
            apoptosed_locations[0][locs_to_empty], apoptosed_locations[1][locs_to_empty]
        ] = EpiType.Empty
        model.epi_intracellular_virus[
            apoptosed_locations[0][locs_to_empty], apoptosed_locations[1][locs_to_empty]
        ] = 0

    # second, spawn to make up for deficiency in the epithelial categories
    # TODO: check the following epithelium state variables
    #  epithelium_ros_damage_counter, epi_regrow_counter, epi_apoptosis_counter,
    #  epi_intracellular_virus, epi_cell_membrane, epi_apoptosis_threshold,
    #  epithelium_apoptosis_counter
    if model.healthy_epithelium_count < desired_healthy_epithelium:
        empty_locations = np.where(model.epithelium == EpiType.Empty)
        assert len(empty_locations) > 0
        healthy_locs = np.random.choice(
            len(empty_locations[1]),
            desired_healthy_epithelium - model.healthy_epithelium_count,
            replace=False,
        )
        model.epithelium[
            empty_locations[0][healthy_locs], empty_locations[1][healthy_locs]
        ] = EpiType.Healthy
        model.epi_intracellular_virus[
            empty_locations[0][healthy_locs], empty_locations[1][healthy_locs]
        ] = 0

    if model.infected_epithelium_count < desired_infected_epithelium:
        empty_locations = np.where(model.epithelium == EpiType.Empty)
        locs_to_infect = np.random.choice(
            len(empty_locations[1]),
            desired_infected_epithelium - model.infected_epithelium_count,
            replace=False,
        )
        model.epithelium[
            empty_locations[0][locs_to_infect], empty_locations[1][locs_to_infect]
        ] = EpiType.Infected
        model.epi_intracellular_virus[
            empty_locations[0][locs_to_infect], empty_locations[1][locs_to_infect]
        ] = 1
        # TODO: move the recalc of epi_intracellular_virus after this?

    if model.dead_epithelium_count < desired_dead_epithelium:
        empty_locations = np.where(model.epithelium == EpiType.Empty)
        assert len(empty_locations) > 0
        locs_to_make_dead = np.random.choice(
            len(empty_locations[1]),
            desired_dead_epithelium - model.dead_epithelium_count,
            replace=False,
        )
        model.epithelium[
            empty_locations[0][locs_to_make_dead], empty_locations[1][locs_to_make_dead]
        ] = EpiType.Dead
        model.epi_intracellular_virus[
            empty_locations[0][locs_to_make_dead], empty_locations[1][locs_to_make_dead]
        ] = 0

    if model.apoptosed_epithelium_count < desired_apoptosed_epithelium:
        empty_locations = np.where(model.epithelium == EpiType.Empty)
        assert len(empty_locations) > 0
        apoptosed_locs = np.random.choice(
            len(empty_locations[1]),
            desired_apoptosed_epithelium - model.apoptosed_epithelium_count,
            replace=False,
        )
        model.epithelium[
            empty_locations[0][apoptosed_locs], empty_locations[1][apoptosed_locs]
        ] = EpiType.Apoptosed
        model.epi_intracellular_virus[
            empty_locations[0][apoptosed_locs], empty_locations[1][apoptosed_locs]
        ] = 0

    dc_delta = int(
        np.rint(desired_state[state_var_indices["dc_count"]] - model.dc_count)
    )
    if dc_delta > 0:
        for _ in range(dc_delta):
            model.create_dc()
    elif dc_delta < 0:
        # need fewer dcs, kill them randomly
        num_to_kill = min(-dc_delta, model.num_dcs)
        dcs_to_kill = np.random.choice(model.num_dcs, num_to_kill, replace=False)
        dc_idcs = np.where(model.dc_mask)[0]
        model.dc_mask[dc_idcs[dcs_to_kill]] = False
        model.num_dcs -= num_to_kill
        assert model.num_dcs == np.sum(model.dc_mask)

    nk_delta = int(
        np.rint(desired_state[state_var_indices["nk_count"]] - model.nk_count)
    )
    if nk_delta > 0:
        model.create_nk(number=int(nk_delta))
    elif nk_delta < 0:
        # need fewer nks, kill them randomly
        num_to_kill = min(-nk_delta, model.num_nks)
        nks_to_kill = np.random.choice(model.num_nks, num_to_kill, replace=False)
        nk_idcs = np.where(model.nk_mask)[0]
        model.nk_mask[nk_idcs[nks_to_kill]] = False
        model.num_nks -= num_to_kill
        assert model.num_nks == np.sum(model.nk_mask)

    pmn_delta = int(
        np.rint(desired_state[state_var_indices["pmn_count"]] - model.pmn_count)
    )
    if pmn_delta > 0:
        # need more pmns
        # adapted from activated_endo_update
        pmn_spawn_list = list(
            zip(
                *np.where(
                    (
                        model.endothelial_adhesion_counter
                        > model.activated_endo_adhesion_threshold
                    )
                    & (model.endothelial_activation == EndoType.Activated)
                    & (
                        np.random.rand(*model.geometry)
                        < model.activated_endo_pmn_spawn_prob
                    )
                )
            )
        )
        if len(pmn_spawn_list) > 0:
            for loc_idx in np.random.choice(len(pmn_spawn_list), pmn_delta):
                model.create_pmn(
                    location=pmn_spawn_list[loc_idx],
                    age=0,
                    jump_dist=model.activated_endo_pmn_spawn_dist,
                )
        else:
            if verbose:
                print("Nowhere to put desired pmns")
    elif pmn_delta < 0:
        # need fewer pmns, kill them randomly
        num_to_kill = min(-pmn_delta, model.num_pmns)
        pmns_to_kill = np.random.choice(model.num_pmns, num_to_kill, replace=False)
        pmn_idcs = np.where(model.pmn_mask)[0]
        model.pmn_mask[pmn_idcs[pmns_to_kill]] = False
        model.num_pmns -= num_to_kill
        assert model.num_pmns == np.sum(model.pmn_mask)

    macro_delta = int(
        np.rint(desired_state[state_var_indices["macro_count"]] - model.macro_count)
    )
    if macro_delta > 0:
        # need more macrophages, create them as in init in random locations
        for _ in range(macro_delta):
            model.create_macro(
                pre_il1=0,
                pre_il18=0,
                inflammasome_primed=False,
                inflammasome_active=False,
                macro_activation_level=0,
                pyroptosis_counter=0,
                virus_eaten=0,
                cells_eaten=0,
            )
    elif macro_delta < 0:
        # need fewer macrophages, kill them randomly
        num_to_kill = min(-macro_delta, model.num_macros)
        macros_to_kill = np.random.choice(model.num_macros, num_to_kill, replace=False)
        macro_idcs = np.where(model.macro_mask)[0]
        model.macro_mask[macro_idcs[macros_to_kill]] = False
        model.num_macros -= num_to_kill
        assert model.num_macros == np.sum(model.macro_mask)
