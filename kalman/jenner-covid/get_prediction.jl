function get_prediction(begin_time, end_time, prior; stochastic_solutions = true)
    # create history: first index is final time
    ts = [begin_time + dt * (idx - history_size) for idx = 1:history_size]

    while true

        history_sample_arr =
            inv_transform(hcat([rand(prior[idx]) for idx = 1:history_size]...))

        history_interpolator = BasicInterpolators.LinearInterpolator(
            ts,
            eachcol(history_sample_arr),
            BasicInterpolators.NoBoundaries(),
        )

        function history_sample(p, t; idxs = nothing)
            history_eval = history_interpolator(t)
            if typeof(idxs) <: Number
                return history_eval[idxs]
            else
                return history_eval
            end
        end

        initial_condition = history_sample_arr[:, end]
        # tau_T_samp <= max_tau_T
        # initial_condition[end] = min(max_tau_T, initial_condition[end])

        const_params = (sird_noise, state_var_noise, param_noise)

        if stochastic_solutions
            sdde_prob_ic = remake(
                sdde_prob;
                tspan = (begin_time, end_time),
                u0 = initial_condition,
                h = history_sample,
                p = const_params,
            )
            prediction = solve(
                sdde_prob_ic,
                sdde_alg;
                alg_hints = [:stiff],
                saveat = dt,
                isoutofdomain = (u, p, t) -> (any(u .< 0.0)),
            )
        else
            dde_prob_ic = remake(
                dde_prob;
                tspan = (begin_time, end_time),
                u0 = initial_condition,
                h = history_sample,
                p = const_params,
            )
            prediction = solve(
                dde_prob_ic,
                dde_alg;
                alg_hints = [:stiff],
                saveat = dt,
                isoutofdomain = (u, p, t) -> (any(u .< 0.0)),
            )
        end

        if !SciMLBase.successful_retcode(prediction)
            continue
        end

        if any(isnan.(hcat(prediction.u...))) | any(isinf.(hcat(prediction.u...)))
            continue
        end

        if minimum(hcat(prediction.u...)) >= 0.0

            # virtual_population_loss = covid_minimizing_fun_severe(prediction)
            # accept = rand() <= exp(-virtual_population_loss)
            accept = true
            if accept
                return prediction, t -> history_sample(p, t; idxs = nothing)
            else
                println(exp(-virtual_population_loss))
            end
        end
        # otherwise go again
    end
end # get_prediction
