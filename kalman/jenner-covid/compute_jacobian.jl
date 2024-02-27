function compute_numerical_jacobian(
    initial_condition,
    sampled_history_interpolated,
    t_0,
    t_1,
)
    const_params = (sird_noise, state_var_noise, param_noise)
    history_interpolated(p, t; idxs = nothing) =
        sampled_history_interpolated(p, t; idxs = idxs)
    J = zeros(unified_state_space_dimension, unified_state_space_dimension)

    initial_condition = inv_transform(initial_condition)

    for component_idx in eachindex(initial_condition)

        # ensure that +/- h never sends us negative
        h = min(0.1, initial_condition[component_idx] / 2.0)
        if (h == 0.0) | issubnormal(h)
            continue
        end

        ic_plus = copy(initial_condition)
        ic_plus[component_idx] += h
        dde_prob_ic = remake(
            dde_prob;
            tspan = (t_0, t_1),
            u0 = ic_plus,
            h = history_interpolated,
            p = const_params,
        )
        plus_prediction = transform(solve(
            dde_prob_ic,
            dde_alg,
            alg_hints = [:stiff],
            saveat = [t_0, t_1],
            isoutofdomain = (u, p, t) -> (any(u .< 0.0)),
        )(
            t_1,
        ))

        ic_minus = copy(initial_condition)
        ic_minus[component_idx] -= h
        dde_prob_ic = remake(
            dde_prob;
            tspan = (t_0, t_1),
            u0 = ic_minus,
            h = history_interpolated,
            p = const_params,
        )
        minus_prediction = transform(solve(
            dde_prob_ic,
            dde_alg,
            alg_hints = [:stiff],
            saveat = [t_0, t_1],
            isoutofdomain = (u, p, t) -> (any(u .< 0.0)),
        )(
            t_1,
        ))

        nan_on_plus = any(isnan.(plus_prediction))
        nan_on_minus = any(isnan.(minus_prediction))
        if nan_on_plus & nan_on_minus
            # can't compute, use default
            J[component_idx, :] .= 0.0
        elseif nan_on_plus | nan_on_minus

            # compute one sided derivative
            dde_prob_ic = remake(
                dde_prob;
                tspan = (t_0, t_1),
                u0 = initial_condition,
                h = history_interpolated,
                p = const_params,
            )
            zero_prediction = transform(solve(
                dde_prob_ic,
                dde_alg,
                alg_hints = [:stiff],
                saveat = [t_0, t_1],
                isoutofdomain = (u, p, t) -> (any(u .< 0.0)),
            )(
                t_1,
            ))

            if any(isnan.(zero_prediction))
                # can't compute, use default
                J[component_idx, :] .= 0.0
            elseif nan_on_plus
                J[component_idx, :] = (zero_prediction - minus_prediction) / h
            elseif nan_on_minus
                J[component_idx, :] = (plus_prediction - zero_prediction) / h
            end

        else
            # both plus and minus predictions are fine, compute the two sided derivative
            J[component_idx, :] = (plus_prediction - minus_prediction) / (2 * h)
        end
    end

    return J
end
